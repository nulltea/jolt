#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::jolt::subtable::eq;
use crate::poly::dense_interleaved_poly::DenseInterleavedPolynomial;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use crate::poly::spartan_interleaved_poly::SpartanInterleavedPolynomial;
use crate::poly::split_eq_poly::{GruenSplitEqPolynomial, SplitEqPolynomial};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::r1cs::builder::{Constraint, OffsetEqConstraint};
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::small_value::svo_helpers::process_svo_sumcheck_rounds;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, KeccakTranscript, Transcript};
use crate::utils::{mul_0_optimized, uninterleave_bits};
use ark_ff::{AdditiveGroup, Field, UniformRand};
use ark_serialize::*;
use itertools::{chain, concat, interleave, izip, Itertools};
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::env;
use std::marker::PhantomData;
use tokio::io::Split;
use tracing_subscriber::layer;

pub trait Bindable<F: JoltField>: Sync {
    fn bind(&mut self, r: F);
}

/// Batched cubic sumcheck used in grand products
pub trait BatchedCubicSumcheck<F, ProofTranscript>: Bindable<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    fn compute_cubic(&self, eq_poly: &SplitEqPolynomial<F>, previous_round_claim: F) -> UniPoly<F>;
    fn final_claims(&self) -> (F, F);

    #[cfg(test)]
    fn sumcheck_sanity_check(&self, eq_poly: &SplitEqPolynomial<F>, round_claim: F);

    #[tracing::instrument(
        skip_all,
        name = "BatchedCubicSumcheck::prove_sumcheck",
        level = "trace"
    )]
    fn prove_sumcheck(
        &mut self,
        claim: &F,
        eq_poly: &mut SplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, (F, F)) {
        let num_rounds = eq_poly.get_num_vars();

        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _round in 0..num_rounds {
            #[cfg(test)]
            self.sumcheck_sanity_check(eq_poly, previous_claim);
            println!(
                "round: {} eq_poly: E1_len {} E2_len {}",
                _round, eq_poly.E1_len, eq_poly.E2_len
            );

            let cubic_poly = self.compute_cubic(eq_poly, previous_claim);
            let compressed_poly = cubic_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            // derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar();

            r.push(r_j);
            // bind polynomials to verifier's challenge
            self.bind(r_j);
            eq_poly.bind(r_j);

            previous_claim = cubic_poly.evaluate(&r_j);
            cubic_polys.push(compressed_poly);
        }

        #[cfg(test)]
        self.sumcheck_sanity_check(eq_poly, previous_claim);

        debug_assert_eq!(eq_poly.len(), 1);

        (
            SumcheckInstanceProof::new(cubic_polys),
            r,
            self.final_claims(),
        )
    }
}

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
    /// Create a sumcheck proof for polynomial(s) of arbitrary degree.
    ///
    /// Params
    /// - `claim`: Claimed sumcheck evaluation (note: currently unused)
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `polys`: Dense polynomials to combine and sumcheck
    /// - `comb_func`: Function used to combine each polynomial evaluation
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (SumcheckInstanceProof, r_eval_point, final_evals)
    /// - `r_eval_point`: Final random point of evaluation
    /// - `final_evals`: Each of the polys evaluated at `r_eval_point`
    #[tracing::instrument(skip_all, name = "Sumcheck.prove", level = "trace")]
    pub fn prove_arbitrary<Func>(
        claim: &F,
        num_rounds: usize,
        polys: &mut Vec<MultilinearPolynomial<F>>,
        comb_func: Func,
        combined_degree: usize,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>)
    where
        Func: Fn(&[F]) -> F + std::marker::Sync,
    {
        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        #[cfg(test)]
        {
            let total_evals = 1 << num_rounds;
            let mut sum = F::zero();
            for i in 0..total_evals {
                let params: Vec<F> = polys.iter().map(|poly| poly.get_coeff(i)).collect();
                sum += comb_func(&params);
            }
            assert_eq!(&sum, claim, "Sumcheck claim is wrong");
        }

        for _round in 0..num_rounds {
            // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
            // for points {0, ..., |g(x)|}

            let mut eval_points = vec![F::zero(); combined_degree];

            let mle_half = polys[0].len() / 2;

            let accum: Vec<Vec<F>> = (0..mle_half)
                .into_par_iter()
                .map(|poly_term_i| {
                    let mut accum = vec![F::zero(); combined_degree];
                    // TODO(moodlezoup): Optimize
                    let evals: Vec<_> = polys
                        .iter()
                        .map(|poly| {
                            poly.sumcheck_evals(
                                poly_term_i,
                                combined_degree,
                                BindingOrder::HighToLow,
                            )
                        })
                        .collect();
                    for j in 0..combined_degree {
                        let evals_j: Vec<_> = evals.iter().map(|x| x[j]).collect();
                        accum[j] += comb_func(&evals_j);
                    }

                    accum
                })
                .collect();

            eval_points
                .par_iter_mut()
                .enumerate()
                .for_each(|(poly_i, eval_point)| {
                    *eval_point = accum
                        .par_iter()
                        .take(mle_half)
                        .map(|mle| mle[poly_i])
                        .sum::<F>();
                });

            eval_points.insert(1, previous_claim - eval_points[0]);
            let univariate_poly = UniPoly::from_evals(&eval_points);
            let compressed_poly = univariate_poly.compress();
            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // bound all tables to the verifier's challenge
            polys
                .par_iter_mut()
                .for_each(|poly| poly.bind(r_j, BindingOrder::HighToLow));
            previous_claim = univariate_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        let final_evals = polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
    }

    #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::prove_spartan_cubic")]
    pub fn prove_spartan_cubic(
        num_rounds: usize,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        az_bz_cz_poly: &mut SpartanInterleavedPolynomial<F>,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, [F; 3]) {
        let mut r: Vec<F> = Vec::new();
        let mut polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut claim = F::zero();

        for round in 0..num_rounds {
            if round == 0 {
                az_bz_cz_poly
                    .first_sumcheck_round(eq_poly, transcript, &mut r, &mut polys, &mut claim);
            } else {
                az_bz_cz_poly
                    .subsequent_sumcheck_round(eq_poly, transcript, &mut r, &mut polys, &mut claim);
            }
        }

        (
            SumcheckInstanceProof::new(polys),
            r,
            az_bz_cz_poly.final_sumcheck_evals(),
        )
    }

    #[tracing::instrument(skip_all)]
    // A specialized sumcheck implementation with the 0th round unrolled from the rest of the
    // `for` loop. This allows us to pass in `witness_polynomials` by reference instead of
    // passing them in as a single `DensePolynomial`, which would require an expensive
    // concatenation. We defer the actual instantiation of a `DensePolynomial` to the end of the
    // 0th round.
    pub fn prove_spartan_quadratic(
        claim: &F,
        num_rounds: usize,
        poly_A: &mut DensePolynomial<F>,
        witness_polynomials: &[&MultilinearPolynomial<F>],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>) {
        let mut r: Vec<F> = Vec::with_capacity(num_rounds);
        let mut polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
        let mut claim_per_round = *claim;

        /*          Round 0 START         */

        let len = poly_A.len() / 2;
        let trace_len = witness_polynomials[0].len();
        // witness_polynomials
        //     .iter()
        //     .for_each(|poly| debug_assert_eq!(poly.len(), trace_len));

        // We don't materialize the full, flattened witness vector, but this closure
        // simulates it
        let witness_value = |index: usize| {
            if (index / trace_len) >= witness_polynomials.len() {
                F::zero()
            } else {
                witness_polynomials[index / trace_len].get_coeff(index % trace_len)
            }
        };

        let poly = {
            // eval_point_0 = \sum_i A[i] * B[i]
            // where B[i] = witness_value(i) for i in 0..len
            let eval_point_0: F = (0..len)
                .into_par_iter()
                .map(|i| {
                    if poly_A[i].is_zero() || witness_value(i).is_zero() {
                        F::zero()
                    } else {
                        poly_A[i] * witness_value(i)
                    }
                })
                .sum();
            // eval_point_2 = \sum_i (2 * A[len + i] - A[i]) * (2 * B[len + i] - B[i])
            // where B[i] = witness_value(i) for i in 0..len, B[len] = 1, and B[i] = 0 for i > len
            let mut eval_point_2: F = (1..len)
                .into_par_iter()
                .map(|i| {
                    if witness_value(i).is_zero() {
                        F::zero()
                    } else {
                        let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
                        let poly_B_bound_point = -witness_value(i);
                        mul_0_optimized(&poly_A_bound_point, &poly_B_bound_point)
                    }
                })
                .sum();
            eval_point_2 += mul_0_optimized(
                &(poly_A[len] + poly_A[len] - poly_A[0]),
                &(F::from_u8(2) - witness_value(0)),
            );

            let evals = [eval_point_0, claim_per_round - eval_point_0, eval_point_2];
            UniPoly::from_evals(&evals)
        };

        let compressed_poly = poly.compress();
        // append the prover's message to the transcript
        compressed_poly.append_to_transcript(transcript);

        //derive the verifier's challenge for the next round
        let r_i: F = transcript.challenge_scalar();
        r.push(r_i);
        polys.push(compressed_poly);

        // Set up next round
        claim_per_round = poly.evaluate(&r_i);

        // bound all tables to the verifier's challenge
        let (_, mut poly_B) = rayon::join(
            || poly_A.bound_poly_var_top_zero_optimized(&r_i),
            || {
                // Simulates `poly_B.bound_poly_var_top(&r_i)` by
                // iterating over `witness_polynomials`
                // We need to do this because we don't actually have
                // a `DensePolynomial` instance for `poly_B` yet.
                let zero = F::zero();
                let one = [F::one()];
                let W_iter = (0..len).into_par_iter().map(witness_value);
                let Z_iter = W_iter
                    .chain(one.into_par_iter())
                    .chain(rayon::iter::repeatn(zero, len));
                let left_iter = Z_iter.clone().take(len);
                let right_iter = Z_iter.skip(len).take(len);
                let B = left_iter
                    .zip(right_iter)
                    .map(|(a, b)| if a == b { a } else { a + r_i * (b - a) })
                    .collect();
                DensePolynomial::new(B)
            },
        );

        /*          Round 0 END          */

        for _i in 1..num_rounds {
            let poly = {
                let (eval_point_0, eval_point_2) =
                    Self::compute_eval_points_spartan_quadratic(poly_A, &poly_B);

                let evals = [eval_point_0, claim_per_round - eval_point_0, eval_point_2];
                UniPoly::from_evals(&evals)
            };

            let compressed_poly = poly.compress();
            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i: F = transcript.challenge_scalar();

            r.push(r_i);
            polys.push(compressed_poly);

            // Set up next round
            claim_per_round = poly.evaluate(&r_i);

            // bound all tables to the verifier's challenge
            rayon::join(
                || poly_A.bound_poly_var_top_zero_optimized(&r_i),
                || poly_B.bound_poly_var_top_zero_optimized(&r_i),
            );
        }

        let evals = vec![poly_A[0], poly_B[0]];
        drop_in_background_thread(poly_B);

        (SumcheckInstanceProof::new(polys), r, evals)
    }

    #[inline]
    #[tracing::instrument(skip_all, name = "Sumcheck::compute_eval_points_spartan_quadratic")]
    pub fn compute_eval_points_spartan_quadratic(
        poly_A: &DensePolynomial<F>,
        poly_B: &DensePolynomial<F>,
    ) -> (F, F) {
        let len = poly_A.len() / 2;
        (0..len)
            .into_par_iter()
            .map(|i| {
                // eval 0: bound_func is A(low)
                let eval_point_0 = if poly_B[i].is_zero() || poly_A[i].is_zero() {
                    F::zero()
                } else {
                    poly_A[i] * poly_B[i]
                };

                // eval 2: bound_func is -A(low) + 2*A(high)
                let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
                let eval_point_2 = if poly_B_bound_point.is_zero() {
                    F::zero()
                } else {
                    let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
                    mul_0_optimized(&poly_A_bound_point, &poly_B_bound_point)
                };

                (eval_point_0, eval_point_2)
            })
            .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
    }

    #[cfg(test)]
    pub fn simulate_prove_cubic_distributed_chunk_wize(
        claim: &F,
        num_rounds: usize,
        eq_poly: &mut SplitEqPolynomial<F>,
        workers_polys: &mut [DenseInterleavedPolynomial<F>],
        chunk_size_per_worker: usize,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, (F, F)) {
        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        let num_workers = workers_polys.len();
        let worker_rounds = chunk_size_per_worker.log_2(); // num variables of per worker chunk

        // Number of global EQ pairs covered across all workers this round:
        // each worker covers chunk_size_per_worker entries = chunk_size_per_worker/2 EQ pairs, so total is (chunk_size_per_worker/2) * num_workers.
        let mut global_eq_pairs = chunk_size_per_worker * num_workers / 2;

        println!(
            "worker rounds: {} (chunk_size_per_worker={} global_eq_pairs={})",
            worker_rounds, chunk_size_per_worker, global_eq_pairs
        );

        for _round in 0..worker_rounds {
            // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
            // for points {0, ..., |g(x)|}
            println!("worker round {} poly: {}", _round, workers_polys[0].len());

            // let mle_half = workers_polys[0].len() / 2;

            let mut eval_points = workers_polys
                .iter()
                .enumerate()
                .map(|(worker, poly)| {
                    dense_interleaved_sumcheck_evals(
                        poly,
                        eq_poly,
                        global_eq_pairs,
                        worker,
                        num_workers,
                    )
                })
                .reduce(|mut eval_points, eval_points_next| {
                    izip!(eval_points.iter_mut(), eval_points_next).for_each(|(a, b)| *a += b);
                    eval_points
                })
                .unwrap();

            global_eq_pairs /= 2;

            // println!("eval points: {:?}", eval_points);
            // println!("--------------");

            eval_points.insert(1, previous_claim - eval_points[0]);
            let univariate_poly = UniPoly::from_evals(&eval_points);
            let compressed_poly = univariate_poly.compress();
            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // bound all tables to the verifier's challenge
            workers_polys.par_iter_mut().for_each(|poly| poly.bind(r_j));
            eq_poly.bind(r_j);
            previous_claim = univariate_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        let poly_remaining_evals: Vec<Vec<F>> = workers_polys
            .iter()
            .map(|poly| poly.coeffs[..poly.len()].to_vec())
            .collect();

        let mut poly = DenseInterleavedPolynomial::new(interleave_n(
            (0..num_workers)
                .map(|w| poly_remaining_evals[w].clone())
                .collect_vec(),
        ));

        // println!("remaining poly {:?}", poly.coeffs);

        let remaining_rounds = num_rounds - worker_rounds;

        println!("remaining rounds: {}", remaining_rounds);

        for _round in 0..remaining_rounds {
            println!(
                "rem round {} | poly: {}",
                _round,
                poly.len(), // &polys[1].coeffs_as_field_elements()[..polys[1].len()]
            );

            let univariate_poly = BatchedCubicSumcheck::<F, ProofTranscript>::compute_cubic(
                &poly,
                &eq_poly,
                previous_claim,
            );
            let compressed_poly = univariate_poly.compress();
            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // bound all tables to the verifier's challenge
            poly.bind(r_j);
            eq_poly.bind(r_j);
            previous_claim = univariate_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        (
            SumcheckInstanceProof::new(compressed_polys),
            r,
            BatchedCubicSumcheck::<F, ProofTranscript>::final_claims(&poly),
        )
    }

    #[cfg(test)]
    pub fn simulate_prove_cubic_distributed_batch_wize(
        claim: &F,
        num_rounds: usize,
        eq_polys: &mut [SplitEqPolynomial<F>],
        workers_polys: &mut [DenseInterleavedPolynomial<F>],
        batch_size_per_worker: usize,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, (F, F)) {
        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        let num_workers = workers_polys.len();
        let log_num_workers = num_workers.log_2();
        let worker_rounds = if batch_size_per_worker.is_power_of_two() {
            batch_size_per_worker.log_2()
        } else {
            batch_size_per_worker.log_2() - 1
        };

        let remaining_rounds = num_rounds - worker_rounds;
        println!(
            "num_rounds: {} worker_rounds {} remaining_rounds {}",
            num_rounds, worker_rounds, remaining_rounds
        );

        println!(
            "poly: {} eq_poly: [{}]",
            workers_polys[0].len(),
            eq_polys
                .iter()
                .map(|eq| format!("E1_len={} E2_len={}", eq.E1_len, eq.E2_len))
                .collect::<Vec<String>>()
                .join(", ")
        );

        let mut eq_chunk_worker = batch_size_per_worker * 2;

        for _round in 0..worker_rounds {
            // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
            // for points {0, ..., |g(x)|}
            // println!("worker round {} poly: {}", _round, workers_polys[0].len());

            // let mle_half = workers_polys[0].len() / 2;

            println!(
                "round {} polys {:?} eq_poly: [{}]",
                _round,
                workers_polys
                    .iter()
                    .map(|p| p.coeffs[..p.len()].len())
                    .collect_vec(),
                eq_polys
                    .iter()
                    .map(|eq| format!("E1_len={} E2_len={}", eq.E1_len, eq.E2_len))
                    .collect::<Vec<String>>()
                    .join(", ")
            );

            let mut eval_points = workers_polys
                .iter()
                .enumerate()
                .map(|(worker, poly)| {
                    // println!("worker {worker} poly: {:?}", &poly.len());
                    // println!(
                    //     "worker {worker} eq_poly E1_len: {} E2_len: {}",
                    //     &eq_polys[worker].E1_len, &eq_polys[worker].E2_len
                    // );
                    let evals = dense_interleaved_sumcheck_evals(
                        poly,
                        &eq_polys[worker],
                        0,
                        worker,
                        num_workers,
                    );
                    // println!("------");
                    evals
                })
                .reduce(|mut eval_points, eval_points_next| {
                    izip!(eval_points.iter_mut(), eval_points_next).for_each(|(a, b)| *a += b);
                    eval_points
                })
                .unwrap();

            eq_chunk_worker /= 2;

            println!("-------");
            println!("eval points: {:?}", eval_points);
            println!("--------------");

            eval_points.insert(1, previous_claim - eval_points[0]);
            let univariate_poly = UniPoly::from_evals(&eval_points);
            let compressed_poly = univariate_poly.compress();
            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // bound all tables to the verifier's challenge
            workers_polys.par_iter_mut().for_each(|poly| poly.bind(r_j));
            eq_polys.par_iter_mut().for_each(|poly| poly.bind(r_j));
            previous_claim = univariate_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        println!(
            "poly: {} eq_poly: [{}]",
            workers_polys[0].len(),
            eq_polys
                .iter()
                .map(|eq| format!("E1_len={} E2_len={}", eq.E1_len, eq.E2_len))
                .collect::<Vec<String>>()
                .join(", ")
        );

        let poly_remaining_evals: Vec<Vec<F>> = workers_polys
            .iter()
            .map(|poly| poly.coeffs[..poly.len()].to_vec())
            .collect();

        let eq_poly_remaining_evals: Vec<Vec<F>> = eq_polys
            .iter()
            .map(|poly| poly.E2[..poly.E2_len].to_vec())
            .collect();

        let mut poly = DenseInterleavedPolynomial::new(
            (0..num_workers)
                .flat_map(|w| poly_remaining_evals[w].clone())
                .collect_vec(),
        );
        let mut eq_poly = SplitEqPolynomial::new_bound(
            vec![F::ZERO],
            (0..num_workers)
                .flat_map(|w| eq_poly_remaining_evals[w].clone())
                .collect_vec(),
            remaining_rounds,
        );

        println!(
            "remaining eq_poly E1_len: {:?} E2_len: {:?}",
            eq_poly.E1_len,
            eq_poly.E2_len,
            // &eq_poly.E2[..eq_poly.E2_len]
        );

        // TODO: send suffix evals to pad to nearest modulo 4
        println!("remaining poly {:?}", poly.coeffs.len());

        for _round in 0..remaining_rounds {
            // println!(
            //     "rem round {} | poly: {}",
            //     _round,
            //     poly.len(), // &polys[1].coeffs_as_field_elements()[..polys[1].len()]
            // );

            let univariate_poly = BatchedCubicSumcheck::<F, ProofTranscript>::compute_cubic(
                &poly,
                &eq_poly,
                previous_claim,
            );
            let compressed_poly = univariate_poly.compress();
            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // bound all tables to the verifier's challenge
            poly.bind(r_j);
            eq_poly.bind(r_j);
            previous_claim = univariate_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        (
            SumcheckInstanceProof::new(compressed_polys),
            r,
            BatchedCubicSumcheck::<F, ProofTranscript>::final_claims(&poly),
        )
    }
}

fn dense_interleaved_sumcheck_evals<F: JoltField>(
    poly: &DenseInterleavedPolynomial<F>,
    eq_poly: &SplitEqPolynomial<F>,
    global_eq_pairs: usize,
    worker_idx: usize,
    num_workers: usize,
) -> Vec<F> {
    // // if worker_idx == 0 {
    // //     println!(
    // //         "split_eq_poly E1_len {} E2_len {} full {}",
    // //         eq_poly.E1_len,
    // //         eq_poly.E2_len,
    // //         eq_poly.E1_len * eq_poly.E2_len
    // //     );
    // // }
    // let (E1, E2) = split_eq_poly_permute(
    //     eq_poly,
    //     global_eq_pairs,
    //     worker_idx,
    //     num_workers,
    //     // EqSplitAxis::Cols,
    // );
    // let E1_len = E1.len();
    // let E2_len = E2.len();

    // let split_eq_permuted = SplitEqPolynomial {
    //     num_vars: eq_poly.num_vars - 1,
    //     E1: E1.clone(),
    //     E2: E2.clone(),
    //     E1_len,
    //     E2_len,
    // };

    // // if worker_idx == 0 {
    // //     println!(
    // //         "eq_poly (merged) len={}",
    // //         eq_poly.merge().Z.len(),
    // //         // eq_poly.merge().Z
    // //     );
    // // }

    // // println!(
    // //     "worker={} split_eq_permuted: E1_len = {}, E2_len = {}",
    // //     worker_idx, E1_len, E2_len
    // // );
    // assert_eq!(
    //     split_eq_permuted.merge().Z,
    //     custom_eq_permute(&eq_poly.merge().Z, global_eq_pairs, worker_idx, num_workers,),
    // );

    let E1 = &eq_poly.E1;
    let E2 = eq_poly.E2.to_vec();
    let E1_len = E1.len();
    let E2_len = E2.len();

    // We use the Dao-Thaler optimization for the EQ polynomial, so there are two cases we
    // must handle. For details, refer to Section 2.2 of https://eprint.iacr.org/2024/1210.pdf
    let cubic_evals = if eq_poly.E1_len == 1 {
        // If `eq_poly.E1` has been fully bound, we compute the cubic polynomial as we
        // would without the Dao-Thaler optimization, using the standard linear-time
        // sumcheck algorithm.

        // poly.par_chunks(4)
        //     .zip(E2.par_chunks(2))
        poly.coeffs[..poly.len()]
            .chunks(4)
            .zip(E2.chunks(2))
            .map(|(layer_chunk, eq_chunk)| {
                let eq_evals = {
                    let eval_point_0 = eq_chunk[0];
                    let m_eq = eq_chunk[1] - eq_chunk[0];
                    let eval_point_2 = eq_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                };
                // let eq_evals = custom_eq_sumcheck_evals(
                //     &eq_poly.E2,
                //     mle_i,
                //     3,
                //     global_eq_pairs,
                //     worker_idx,
                //     num_workers,
                // );

                let left = (
                    *layer_chunk.first().unwrap_or(&F::zero()),
                    *layer_chunk.get(2).unwrap_or(&F::zero()),
                );
                let right = (
                    *layer_chunk.get(1).unwrap_or(&F::zero()),
                    *layer_chunk.get(3).unwrap_or(&F::zero()),
                );

                let m_left = left.1 - left.0;
                let m_right = right.1 - right.0;

                let left_eval_2 = left.1 + m_left;
                let left_eval_3 = left_eval_2 + m_left;

                let right_eval_2 = right.1 + m_right;
                let right_eval_3 = right_eval_2 + m_right;

                // println!(
                //     "partial evals: {:?}",
                //     [
                //         [eq_evals.0, eq_evals.1, eq_evals.2],
                //         [left.0, left_eval_2, left_eval_3],
                //         [right.0, right_eval_2, right_eval_3]
                //     ]
                // );

                (
                    eq_evals.0 * left.0 * right.0,
                    eq_evals.1 * left_eval_2 * right_eval_2,
                    eq_evals.2 * left_eval_3 * right_eval_3,
                )
            })
            .reduce(
                // || (F::zero(), F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
            )
            .unwrap()
    } else {
        // If `eq_poly.E1` has NOT been fully bound, we compute the cubic polynomial
        // using the nested summation approach described in Section 2.2 of https://eprint.iacr.org/2024/1210.pdf
        //
        // Note, however, that we reverse the inner/outer summation compared to the
        // description in the paper. I.e. instead of:
        //
        // \sum_x1 ((1 - j) * E1[0, x1] + j * E1[1, x1]) * (\sum_x2 E2[x2] * \prod_k ((1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2)))
        //
        // we do:
        //
        // \sum_x2 E2[x2] * (\sum_x1 ((1 - j) * E1[0, x1] + j * E1[1, x1]) * \prod_k ((1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2)))
        //
        // because it has better memory locality.

        // We start by computing the E1 evals:
        // (1 - j) * E1[0, x1] + j * E1[1, x1]
        let E1_evals: Vec<_> = E1[..E1_len]
            .par_chunks(2)
            .map(|E1_chunk| {
                let eval_point_0 = E1_chunk[0];
                let m_eq = E1_chunk[1] - E1_chunk[0];
                let eval_point_2 = E1_chunk[1] + m_eq;
                let eval_point_3 = eval_point_2 + m_eq;
                (eval_point_0, eval_point_2, eval_point_3)
            })
            .collect();

        let chunk_size = (poly.len().next_power_of_two() / E2_len).max(1);
        E2[..E2_len]
            .par_iter()
            .zip(poly.par_chunks(chunk_size))
            .map(|(E2_eval, P_x2)| {
                // The for-loop below corresponds to the inner sum:
                // \sum_x1 ((1 - j) * E1[0, x1] + j * E1[1, x1]) * \prod_k ((1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2))
                let mut inner_sum = (F::zero(), F::zero(), F::zero());
                for (E1_evals, P_chunk) in E1_evals.iter().zip(P_x2.chunks(4)) {
                    let left = (
                        *P_chunk.first().unwrap_or(&F::zero()),
                        *P_chunk.get(2).unwrap_or(&F::zero()),
                    );
                    let right = (
                        *P_chunk.get(1).unwrap_or(&F::zero()),
                        *P_chunk.get(3).unwrap_or(&F::zero()),
                    );
                    let m_left = left.1 - left.0;
                    let m_right = right.1 - right.0;

                    let left_eval_2 = left.1 + m_left;
                    let left_eval_3 = left_eval_2 + m_left;

                    let right_eval_2 = right.1 + m_right;
                    let right_eval_3 = right_eval_2 + m_right;

                    inner_sum.0 += E1_evals.0 * left.0 * right.0;
                    inner_sum.1 += E1_evals.1 * left_eval_2 * right_eval_2;
                    inner_sum.2 += E1_evals.2 * left_eval_3 * right_eval_3;
                }

                // Multiply the inner sum by E2[x2]
                (
                    *E2_eval * inner_sum.0,
                    *E2_eval * inner_sum.1,
                    *E2_eval * inner_sum.2,
                )
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
            )
    };

    vec![cubic_evals.0, cubic_evals.1, cubic_evals.2]
}

pub fn split_eq_poly_permute<F: JoltField>(
    eq_poly: &SplitEqPolynomial<F>,
    global_eq_pairs: usize,
    worker: usize,
    num_workers: usize,
) -> (Vec<F>, Vec<F>) {
    debug_assert!(num_workers >= 2 && num_workers.is_power_of_two());
    debug_assert!(global_eq_pairs >= num_workers);
    debug_assert_eq!(global_eq_pairs % num_workers, 0);

    let e1_len = eq_poly.E1_len;
    let e2_len = eq_poly.E2_len;
    debug_assert!(e1_len > 0 && e2_len > 0);

    // Degenerate 1D case: just reuse custom_eq_permute on E2.
    if e1_len == 1 {
        let eq_evals = &eq_poly.E2[..e2_len];
        let total = eq_evals.len();
        debug_assert_eq!(total % 2, 0, "eq_evals length must be even");
        let num_pairs_total = total / 2;

        assert!(
            num_pairs_total % global_eq_pairs == 0,
            "num_pairs_total must be divisible by global_eq_pairs"
        );

        let block = global_eq_pairs / num_workers;
        let cols = num_pairs_total / global_eq_pairs;
        let pairs_per_worker = num_pairs_total / num_workers;

        let mut out_e2 = Vec::with_capacity(pairs_per_worker * 2);
        for k in 0..cols {
            let base = (k * num_workers + worker) * block;
            for off in 0..block {
                let g = base + off;
                let idx = 2 * g;
                out_e2.push(eq_evals[idx]);
                out_e2.push(eq_evals[idx + 1]);
            }
        }
        debug_assert_eq!(out_e2.len(), pairs_per_worker * 2);
        let out_e1 = eq_poly.E1[..1].to_vec();
        return (out_e1, out_e2);
    }

    // General 2D factorized case.
    let total = e1_len * e2_len;
    debug_assert_eq!(total % 2, 0, "eq table length must be even");
    let num_pairs_total = total / 2;
    assert!(
        num_pairs_total % global_eq_pairs == 0,
        "num_pairs_total must be divisible by global_eq_pairs"
    );

    let block = global_eq_pairs / num_workers; // pairs per block
    let cols = num_pairs_total / global_eq_pairs; // repetitions
    let pairs_per_worker = num_pairs_total / num_workers;
    let expected_len = pairs_per_worker * 2;

    // Collect which rows and columns this worker actually touches (based on idx = 2g).
    let mut row_set: BTreeSet<usize> = BTreeSet::new();
    let mut col_even: BTreeSet<usize> = BTreeSet::new();

    for k in 0..cols {
        for off in 0..block {
            let g = (k * num_workers + worker) * block + off;
            let idx = 2 * g; // first index of (2g,2g+1)
            let row = idx / e1_len;
            let col = idx % e1_len;
            row_set.insert(row);
            if row == 0 {
                col_even.insert(col);
            }
        }
    }

    // Columns from row 0: each pair contributes (c, c+1).
    let mut col_set: BTreeSet<usize> = BTreeSet::new();
    for c in col_even {
        col_set.insert(c);
        let c1 = c + 1;
        assert!(c1 < e1_len, "pair (2g,2g+1) crosses row boundary");
        col_set.insert(c1);
    }

    let all_rows_len = e2_len;
    let all_cols_len = e1_len;
    let row_full = row_set.len() == all_rows_len;
    let col_full = col_set.len() == all_cols_len;
    let col_nonempty = !col_set.is_empty();

    // Case A: column split (all rows, subset of columns).
    if row_full && col_nonempty && col_set.len() < all_cols_len {
        let cols_for_worker: Vec<usize> = col_set.into_iter().collect();
        let e1_len_worker = cols_for_worker.len();
        debug_assert_eq!(
            expected_len % e1_len_worker,
            0,
            "worker chunk len {} not multiple of E1_len_worker {}",
            expected_len,
            e1_len_worker
        );
        let e2_len_worker = expected_len / e1_len_worker;
        debug_assert_eq!(
            e2_len_worker, e2_len,
            "Cols split: expected to keep E2_len {}, got {}",
            e2_len, e2_len_worker
        );

        let mut E1_new = Vec::with_capacity(e1_len_worker);
        for c in cols_for_worker {
            E1_new.push(eq_poly.E1[c]);
        }
        let mut E2_new = Vec::with_capacity(e2_len_worker);
        E2_new.extend_from_slice(&eq_poly.E2[..e2_len_worker]);

        return (E1_new, E2_new);
    }

    // Case B: row split (subset of rows, all columns).
    if row_set.len() < all_rows_len && (col_set.is_empty() || col_full) {
        let mut rows_for_worker: Vec<usize> = row_set.into_iter().collect();
        rows_for_worker.sort();
        let e2_len_worker = rows_for_worker.len();
        debug_assert_eq!(
            expected_len % e1_len,
            0,
            "worker chunk len {} not multiple of E1_len {}",
            expected_len,
            e1_len
        );
        debug_assert_eq!(
            expected_len / e1_len,
            e2_len_worker,
            "Rows split: expected E2_len_worker {} from permutation, got {}",
            expected_len / e1_len,
            e2_len_worker
        );

        let mut E2_new = Vec::with_capacity(e2_len_worker);
        for r in rows_for_worker {
            E2_new.push(eq_poly.E2[r]);
        }
        let E1_new = eq_poly.E1[..e1_len].to_vec();

        return (E1_new, E2_new);
    }

    // Anything else would mix rows and columns in a non-separable way.
    panic!(
        "split_eq_poly_permute: unsupported worker pattern (rows={}, cols={})",
        row_set.len(),
        col_set.len()
    );
}

/// Compute per-worker delta in *elements* for given batch_size, num_workers (power of 2),
/// and chunk_size = 2^L (elements per chunk).
/// Uses:
///   N_worker = floor(N / W)
///   B = N_worker * 2^(L-2)
///   wr = floor(log2(B)) - 1
///   M = 4 * 2^wr = 2^(t + L - 1), where t = floor(log2 N_worker)
///   P_layer = N_worker * chunk_size
///   delta = P_layer mod M, then:
///     if (P_layer + delta) / 2^wr ≡ 0 (mod 4) → +delta
///     else                                    → -delta
pub fn calculate_delta_per_worker(
    batch_size: usize,
    num_workers: usize,
    chunk_size: usize,
) -> isize {
    assert!(num_workers > 0 && num_workers.is_power_of_two());
    assert!(chunk_size > 0 && chunk_size.is_power_of_two());

    // N_worker = floor(N / W)
    let n_worker = batch_size / num_workers;
    if n_worker == 0 {
        return 0;
    }

    // L = log2(chunk_size)
    let L = chunk_size.trailing_zeros() as u32;

    // t = floor(log2(N_worker))
    let t = (usize::BITS - 1 - n_worker.leading_zeros()) as u32;

    // M = 2^(t + L - 1)
    let m_exp = t + L - 1;
    assert!(m_exp < 127, "exponent too large for u128");
    let M: u128 = 1u128 << m_exp;
    let mask: u128 = M - 1;

    // P_layer = N_worker * chunk_size
    let p_layer: u128 = (n_worker as u128) * (chunk_size as u128);

    // base delta (in elements)
    let delta_base: u128 = p_layer & mask; // P_layer mod M

    if delta_base == 0 {
        return 0;
    }

    let half_M: u128 = M >> 1;

    // Check +delta branch:
    // ((P_layer + delta) / 2^wr) mod 4 == 0  <=>  P_layer + delta ≡ 0 mod M
    // which holds iff delta ≡ (-P_layer) mod M. Our candidate delta_base already
    // satisfies P_layer + delta_base ≡ 0 mod M, so +delta is valid iff delta_base == M/2.
    if delta_base == half_M {
        delta_base as isize // use +delta
    } else {
        -(delta_base as isize) // fall back to -delta
    }
}

/// Given:
/// - num_memories = M (each memory = 2 chunks),
/// - num_workers = W (power of 2),
/// - chunk_size = C (elements per chunk, power of 2),
/// split the big polynomial [0 .. 2*M*C) among workers like in your example,
/// and return which memories this `worker_idx` touches plus the shared `delta`.
pub fn read_write_memories_for_worker(
    num_memories: usize,
    chunk_size: usize,
    num_workers: usize,
    worker_idx: usize,
) -> (Vec<usize>, isize) {
    assert!(num_memories > 0);
    assert!(num_workers > 0 && num_workers.is_power_of_two());
    assert!(chunk_size > 0 && chunk_size.is_power_of_two());
    assert!(worker_idx < num_workers);

    // Total chunks and per-worker baseline
    let n_chunks = 2 * num_memories; // N = M * 2
    let n_worker = n_chunks / num_workers; // floor(N/W)

    // Shared delta in *elements*
    let delta = calculate_delta_per_worker(n_chunks, num_workers, chunk_size);

    // Length (in elements) of a non-last worker's portion
    let base_len = (n_worker as isize * chunk_size as isize + delta) as usize;
    assert!(base_len > 0, "non-last worker chunk_len must be positive");

    // Total elements in the full poly
    let total_elems = n_chunks * chunk_size;

    // Compute this worker's element range [start, end)
    let (start_elem, end_elem) = if worker_idx + 1 < num_workers {
        let start = base_len * worker_idx;
        let end = start + base_len;
        (start, end)
    } else {
        // last worker gets the remainder
        let start = base_len * (num_workers - 1);
        let end = total_elems;
        (start, end)
    };

    // Each memory i occupies [i * 2*C, i * 2*C + 2*C)
    let mem_span = 2 * chunk_size;
    let mut memories = Vec::new();
    for mem_idx in 0..num_memories {
        let mem_start = mem_idx * mem_span;
        let mem_end = mem_start + mem_span;
        // non-empty intersection with [start_elem, end_elem)
        if mem_start < end_elem && mem_end > start_elem {
            memories.push(mem_idx);
        }
    }

    (memories, delta)
}

/// For a given worker, return which memories (grouped by subtable) fall into its
/// polynomial slice, when the full poly is built as:
///   for each subtable:
///       [header_chunk] + [mem_chunk_0] + [mem_chunk_1] + ...
/// Each chunk has `chunk_size` elements.
/// Splitting is *by elements* using the delta trick:
///   - N_blocks = total headers + total memories
///   - N_chunks = N_blocks  (1 chunk per block)
///   - All workers except last get len = N_worker * chunk_size + delta elements
///   - Last worker gets the remaining elements.
/// A memory’s chunk may be split between workers; we assign the memory to a worker
/// iff that worker’s element range intersects that memory’s chunk.
pub fn init_final_subtables_for_worker(
    subtable_to_memory_indices: &[Vec<usize>],
    chunk_size: usize,  // e.g. 1 << 16
    num_workers: usize, // power of two
    worker_idx: usize,
) -> (
    Vec<(usize /*subtable_idx*/, Vec<usize> /*memories*/)>,
    isize, /*delta*/
) {
    assert!(num_workers > 0 && num_workers.is_power_of_two());
    assert!(chunk_size > 0 && chunk_size.is_power_of_two());
    assert!(worker_idx < num_workers);

    // Build prefix sums in *block* space: each subtable contributes
    // 1 header block + len(subtable) memory blocks.
    let mut pref = Vec::with_capacity(subtable_to_memory_indices.len() + 1);
    pref.push(0usize);
    for st in subtable_to_memory_indices {
        pref.push(pref.last().copied().unwrap() + 1 + st.len());
    }
    let total_blocks = *pref.last().unwrap();
    if total_blocks == 0 {
        panic!("No blocks to process")
    }

    // Total "chunks" = total_blocks (1 chunk per block).
    let batch_size = total_blocks; // N
    assert!(
        batch_size >= num_workers,
        "not enough blocks to sensibly split across workers"
    );

    // N_worker = floor(N / W)
    let n_worker = batch_size / num_workers;

    // Delta in *elements* (may be positive or negative).
    let delta_elems = calculate_delta_per_worker(batch_size, num_workers, chunk_size);

    // Length of a non-last worker's slice, in elements.
    let base_len_elems = (n_worker as isize * chunk_size as isize + delta_elems) as usize;
    assert!(base_len_elems > 0, "non-last worker slice must be positive");

    // Total elements in the full polynomial.
    let total_elems = batch_size * chunk_size;

    // Element range [start_elem, end_elem) for this worker.
    let (start_elem, end_elem) = if worker_idx + 1 < num_workers {
        let start = base_len_elems * worker_idx;
        let end = start + base_len_elems;
        (start, end)
    } else {
        let start = base_len_elems * (num_workers - 1);
        let end = total_elems;
        (start, end)
    };

    // Map that element interval back to per-subtable memories.
    // Block k corresponds to element range [k * chunk_size, (k + 1) * chunk_size).
    // For subtable i:
    //   header block index = pref[i]
    //   memory j (0..st.len()) block index = pref[i] + 1 + j
    let mut out: Vec<(usize, Vec<usize>)> = Vec::new();

    for (i, st) in subtable_to_memory_indices.iter().enumerate() {
        let st_beg_block = pref[i];
        let st_end_block = pref[i + 1];
        // If even the header is beyond this worker's range, we can break.
        let st_beg_elem = st_beg_block * chunk_size;
        if st_beg_elem >= end_elem {
            break;
        }

        let mut mems_for_worker = Vec::new();
        let mems_beg_block = st_beg_block + 1;
        for (j, &mem_id) in st.iter().enumerate() {
            let block_idx = mems_beg_block + j;
            let block_start = block_idx * chunk_size;
            let block_end = block_start + chunk_size;

            if block_start >= end_elem {
                break; // no further mems from this subtable can intersect
            }
            if block_end > start_elem {
                // Non-empty intersection with worker's [start_elem, end_elem)
                mems_for_worker.push(mem_id);
            }
        }

        if !mems_for_worker.is_empty() {
            out.push((i, mems_for_worker));
        }
    }

    (out, delta_elems)
}

#[test]
fn test_memories_allocation() {
    type F = ark_bn254::Fr;
    const NUM_MEMORIES: usize = 51;
    const M: usize = 1 << 16;

    let chunk_size: usize = env::var("CHUNK_SIZE")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap();

    let W: usize = env::var("NUM_WORKERS")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .unwrap();

    assert!(
        chunk_size.is_power_of_two(),
        "chunk_size must be a power of two"
    );
    assert!(W.is_power_of_two(), "num_workers must be a power of two");

    println!("NUM_WORKERS={} | CHUNK_SIZE={}", W, chunk_size);

    let W_log2 = W.log_2();
    // let leaves_len = chunk_size * N;
    // let num_layers = (leaves_len / N).log_2();
    // println!("num_layers: {}", num_layers);

    let N_rw = NUM_MEMORIES * 2;

    // let mut in_read_write = vec![];

    // for i in 1..N_rw + 1 {
    //     in_read_write.extend(vec![F::from(i as u64); chunk_size]);
    // }
    // println!("in_interleaved: {:?}", in_interleaved);

    let subtable_to_memory_indices: Vec<Vec<usize>> = vec![
        vec![0, 1, 2, 3],
        vec![4],
        vec![5, 6, 7, 8],
        vec![9],
        vec![10],
        vec![11, 12, 13, 14],
        vec![15],
        vec![16, 17, 18, 19],
        vec![20, 21, 22, 23],
        vec![24],
        vec![25],
        vec![26],
        vec![27],
        vec![28],
        vec![29],
        vec![30],
        vec![31],
        vec![32],
        vec![33],
        vec![34, 35, 36, 37],
        vec![38, 39, 40, 41],
        vec![42, 43, 44, 45],
        vec![46, 47, 48, 49],
        vec![50],
    ];

    let mut read_index = 1u64;
    let read_cts = (0..NUM_MEMORIES)
        .map(|_| {
            let read_ct = vec![F::from(read_index); chunk_size];
            read_index += 2;
            read_ct
        })
        .collect_vec();
    // println!("read_cts: {:?}", read_cts);

    let mut init_index = 1u64;
    let (materialized_subtables, final_cts): (Vec<_>, Vec<_>) = subtable_to_memory_indices
        .iter()
        .map(|memories| {
            let subtable = vec![F::from(init_index); M];
            init_index += 1;
            let final_cts = (0..memories.len())
                .map(|mi| vec![F::from(init_index + mi as u64); M])
                .collect_vec();
            init_index += memories.len() as u64;

            // let subtable = vec![F::from(0); M];
            // let final_cts = memories
            //     .iter()
            //     .map(|mi| vec![F::from(*mi as u64 + 1); M])
            //     .collect_vec();

            (subtable, final_cts)
        })
        .unzip();
    // izip!(&materialized_subtables, &final_cts)
    //     .for_each(|(subtable, final_cts)| println!("init: {:?} final: {:?}", subtable, final_cts));
    let final_cts = final_cts.into_iter().flatten().collect_vec();

    println!("\n/---------- Construct layers ----------/");

    let mut read_write_workers = vec![vec![]; W];
    let mut init_final_workers = vec![vec![]; W];

    for w in 0..W {
        let (worker_rw_memories, delta) =
            read_write_memories_for_worker(NUM_MEMORIES, chunk_size, W, w);
        let w_chunk_len = worker_rw_memories.len() * 2 * chunk_size;

        println!(
            "worker {} read_write_memories [{}]: {:?} delta: {} w_chunk_len: {}",
            w,
            worker_rw_memories.len(),
            worker_rw_memories,
            delta,
            w_chunk_len
        );
        read_write_workers[w] = worker_rw_memories
            .into_iter()
            .flat_map(|memory_index| {
                let read_cts = &read_cts[memory_index];

                let read_fingerprints: Vec<_> = (0..chunk_size).map(|i| read_cts[i]).collect();
                let write_fingerprints: Vec<_> = read_fingerprints
                    .iter()
                    .map(|read_fingerprint| *read_fingerprint + F::ONE)
                    .collect();

                [read_fingerprints, write_fingerprints]
            })
            .collect::<Vec<_>>();

        println!("---------");

        let (worker_memories_for_subtables, delta) =
            init_final_subtables_for_worker(&subtable_to_memory_indices, M, W, w);
        let final_memories = worker_memories_for_subtables
            .iter()
            .flat_map(|(_, memories)| memories)
            .copied()
            .collect_vec();
        println!(
            "worker {} memories_for_subtables [{}]: {:?} delta: {}",
            w,
            final_memories.len(),
            worker_memories_for_subtables,
            delta
        );

        init_final_workers[w] = worker_memories_for_subtables
            .into_iter()
            .flat_map(|(subtable_index, memories)| {
                let has_init = subtable_to_memory_indices[subtable_index][0] == memories[0];
                let subtable = &materialized_subtables[subtable_index];
                let mut leaves_len = M * memories.len();
                if has_init {
                    leaves_len += M;
                }
                let mut leaves = vec![F::ZERO; leaves_len];
                let mut leaf_index = 0;

                // Init leaves
                if has_init {
                    (0..M).for_each(|i| {
                        leaves[i] = subtable[i];
                    });
                    leaf_index = M;
                }

                // Final leaves
                for memory_index in memories {
                    (0..M).for_each(|i| {
                        leaves[leaf_index] = final_cts[memory_index][i];
                        leaf_index += 1;
                    });
                }

                leaves
            })
            .collect();

        println!("------------------");
    }

    // for worker_index in 0..W {
    //     println!(
    //         "worker: {} read_write {:?}",
    //         worker_index, read_write_workers[worker_index]
    //     );
    // }

    assert_eq!(
        read_write_workers
            .iter()
            .map(|wp| wp.iter().flatten().collect_vec().len())
            .sum::<usize>(),
        102 * chunk_size
    );

    // for worker_index in 0..W {
    //     println!(
    //         "worker: {} init_final {:?}",
    //         worker_index, init_final_workers[worker_index]
    //     );
    // }

    assert_eq!(
        init_final_workers.iter().map(|wp| wp.len()).sum::<usize>(),
        75 * M
    );
}

#[cfg(test)]
fn run_simulation_dbgp_batch_wize<F: JoltField>(chunk_size: usize, N: usize, W: usize) {
    assert!(
        chunk_size.is_power_of_two(),
        "chunk_size must be a power of two"
    );
    assert!(W.is_power_of_two(), "num_workers must be a power of two");
    let N_worker = N / W; // floor(N / W)

    println!(
        "NUM_WORKERS={} | CHUNK_SIZE={} | BATCH_SIZE={}={}/worker",
        W, chunk_size, N, N_worker
    );

    let W_log2 = W.log_2();
    let leaves_len = chunk_size * N;
    let num_layers = (leaves_len / N).log_2();
    println!("num_layers: {}", num_layers);

    let mut in_interleaved = vec![];

    for i in 1..N + 1 {
        in_interleaved.extend(vec![F::from(i as u64); chunk_size]);
    }
    println!("in_interleaved: {:?}", in_interleaved);

    #[derive(Debug, Clone)]
    struct LayerCircuit<F: JoltField> {
        layer_idx: usize,
        polys: Vec<DenseInterleavedPolynomial<F>>,
    }

    struct LayerProof<F: JoltField> {
        proof: SumcheckInstanceProof<F, KeccakTranscript>,
        left_claim: F,
        right_claim: F,
    }

    println!("\n/---------- Construct layers ----------/");

    let mut w_interleaved = vec![vec![]; W];
    let mut leaves = in_interleaved.clone();
    let delta = calculate_delta_per_worker(N, W, chunk_size);
    println!("batch_size_worker: {} delta: {}", N_worker, delta);
    let num_memories = 51;
    for w in 0..W {
        let (worker_memories, _delta) =
            read_write_memories_for_worker(num_memories, chunk_size, W, w);
        println!(
            "worker_memories [{}]: {:?} delta: {}",
            worker_memories.len(),
            worker_memories,
            _delta
        );

        let mut w_chunk_len = ((chunk_size * N_worker) as isize + delta) as usize;
        if w == W - 1 {
            w_chunk_len = N * chunk_size - w_chunk_len * (W - 1);
        };
        println!("worker: {} w_chunk_len: {}", w, w_chunk_len);

        w_interleaved[w] = leaves.drain(0..w_chunk_len).collect();
    }
    assert!(leaves.is_empty());

    let input_layer = LayerCircuit {
        layer_idx: num_layers,
        polys: w_interleaved
            .iter()
            .cloned()
            .map(DenseInterleavedPolynomial::new)
            .collect(),
    };

    println!(
        "input layer ({}) | polys: {:?}",
        input_layer.layer_idx,
        input_layer
            .polys
            .iter()
            .map(|p| &p.coeffs)
            .collect::<Vec<_>>()
    );

    let worker_num_layers = num_layers - 2 - 1;
    let coordinator_num_layers = 1 + 1;
    println!(
        "worker layers: {} coordinator layers: {}",
        worker_num_layers, coordinator_num_layers
    );

    let mut worker_layers = vec![input_layer];

    for i in 0..worker_num_layers {
        let prev_layer = &worker_layers[i];
        let layer_idx = num_layers - i - 1;

        let polys = prev_layer
            .polys
            .par_iter()
            .map(DenseInterleavedPolynomial::layer_output)
            .collect::<Vec<_>>();
        println!(
            "worker layer {} | out: {:?}",
            layer_idx,
            polys.iter().map(|p| &p.coeffs).collect::<Vec<_>>()
        );
        println!("-------------");

        let next_layer = LayerCircuit { layer_idx, polys };

        worker_layers.push(next_layer);
    }

    // Next levels are constructed/run by coordinator.
    // Alternatively, if we allow cross-worker communication, we can construct/run them by workers:
    // - At each subsequent layer, we can have workers send their partial results to smaller subnet (half the size)
    // - The proving is also done in this fashion.
    let mut coordinator_layers: Vec<LayerCircuit<F>> = Vec::with_capacity(coordinator_num_layers);

    let mut switched = false;
    for i in 0..coordinator_num_layers {
        let layer_idx = num_layers - worker_num_layers - i - 1;

        let next_layer_poly = if !switched {
            let prev_layer = worker_layers.last().unwrap();

            // What coordinator receives from workers (sub hashes)
            let next_layer_coeffs = prev_layer
                .polys
                .par_iter()
                .flat_map(|poly| {
                    let (left, right) = poly.uninterleave();
                    izip!(&left, &right).map(|(a, b)| *a * *b).collect_vec()
                })
                .collect::<Vec<_>>();

            println!("-------------");
            println!(
                "worker layer {} | next_layer_coeffs: {:?}",
                num_layers - worker_num_layers,
                next_layer_coeffs
            );
            switched = true;
            DenseInterleavedPolynomial::new(next_layer_coeffs)
        } else {
            // After switch run normally
            let prev_layer = coordinator_layers.last().unwrap();

            prev_layer.polys[0].layer_output()
        };
        println!("-------------");

        let next_layer = LayerCircuit {
            layer_idx,
            polys: vec![next_layer_poly],
        };

        // println!(
        //     "coordinator layer {} | poly {:?}",
        //     layer_idx, next_layer.polys[0].coeffs
        // );
        coordinator_layers.push(next_layer);
    }

    let grand_product_output = {
        let last_layer = coordinator_layers.last().unwrap();
        let (left, right) = last_layer.polys[0].uninterleave();
        izip!(left, right).map(|(a, b)| a * b).collect::<Vec<_>>()
    };

    // let grand_product_output = {
    //     let last_layer = worker_layers.last().unwrap();
    //     last_layer
    //         .polys
    //         .par_iter()
    //         .flat_map(|poly| {
    //             let (left, right) = poly.uninterleave();
    //             izip!(&left, &right).map(|(a, b)| *a * *b).collect_vec()
    //         })
    //         .collect::<Vec<_>>()
    // };

    println!("grand product output {:?}", grand_product_output);

    let mut layer_proofs = vec![];
    let mut transcript = KeccakTranscript::new(&[]);

    println!("\n/---------- Coordinator prover ----------/");

    //------ Output layer (N) prover
    let output_mle = DensePolynomial::new_padded(grand_product_output.clone());
    let mut num_rounds = output_mle.get_num_vars();
    let mut r_grand_product: Vec<_> = transcript.challenge_vector(num_rounds);
    // gkr output claim, will be updated as output claim of each subsequent layer as we progress to input layer
    let mut grand_product_claim = output_mle.evaluate(&r_grand_product);

    // GKR proves from output layer to input layer
    for mut layer in coordinator_layers.into_iter().rev() {
        println!("layer {} rounds: {}", layer.layer_idx, num_rounds);

        let mut eq_poly = SplitEqPolynomial::new(&r_grand_product);

        let (proof, r_sumcheck, (left_claim, right_claim)) =
            layer.polys[0].prove_sumcheck(&grand_product_claim, &mut eq_poly, &mut transcript);

        layer_proofs.push(LayerProof {
            proof,
            left_claim,
            right_claim,
        });

        let r_layer = transcript.challenge_scalar();
        grand_product_claim = left_claim + r_layer * (right_claim - left_claim);

        println!("layer {} claim: {:?}", layer.layer_idx, grand_product_claim);

        r_grand_product = r_sumcheck.iter().rev().copied().collect();
        r_grand_product.push(r_layer); // pass r_grand_product to next layer
        num_rounds += 1;
    }

    //------ Distributed layers prover

    println!("\n/------------ Worker prover ------------/");

    let mut batch_size_per_worker = 2 * N_worker;
    for mut layer in worker_layers.iter().cloned().rev() {
        // println!(
        //     "layer {} rounds: {:?} polys: {:?}",
        //     layer.layer_idx,
        //     num_rounds,
        //     layer.polys.iter().map(|p| &p.coeffs).collect::<Vec<_>>()
        // );
        let eq_pairs_per_worker = layer.polys[0].len() / 2;

        let mut eq_polys = (0..W)
            .map(|w| {
                SplitEqPolynomial::new_chunk_custom(
                    &r_grand_product,
                    W_log2,
                    w,
                    eq_pairs_per_worker,
                )
            })
            .collect_vec();

        let (proof, r_sumcheck, (left_claim, right_claim)) =
            SumcheckInstanceProof::<F, KeccakTranscript>::simulate_prove_cubic_distributed_batch_wize(
                &grand_product_claim,
                num_rounds,
                &mut eq_polys,
                &mut layer.polys,
                batch_size_per_worker,
                &mut transcript,
            );

        layer_proofs.push(LayerProof {
            proof,
            left_claim,
            right_claim,
        });

        let r_layer = transcript.challenge_scalar();
        grand_product_claim = left_claim + r_layer * (right_claim - left_claim);

        println!("layer {} claim: {:?}", layer.layer_idx, grand_product_claim);

        r_grand_product = r_sumcheck.iter().rev().copied().collect();
        r_grand_product.push(r_layer); // pass r_grand_product2 to next layer
        num_rounds += 1;
        batch_size_per_worker *= 2;
        // eq_pairs_per_worker *= 2;
        println!("-------------");
    }

    // Compute openings
    println!("\n/---------- Compute openings ----------/");

    let (_, r_opening) =
        r_grand_product.split_at(grand_product_output.len().next_power_of_two().log_2());
    // let (r_opening_worker, r_opening_remaining) =
    //     r_opening.split_at(chunk_size_worker.next_power_of_two().log_2());
    // println!(
    //     "r_grand_product {:?} r_opening {:?} r_opening_worker {:?} r_opening_remaining {:?}",
    //     r_grand_product.len(),
    //     r_opening.len(),
    //     r_opening_worker.len(),
    //     r_opening_remaining.len()
    // );

    let prover_openings: Vec<_> = worker_layers[0]
        .polys
        .iter()
        .flat_map(|poly| {
            let w_opennings = poly
                .coeffs
                .chunks(chunk_size)
                .map(|w_chunk| MultilinearPolynomial::from(w_chunk.to_vec()).evaluate(&r_opening))
                .collect_vec();
            // assert_eq!(w_opennings.len(), N_worker);
            w_opennings
        })
        .collect();
    // .fold(vec![vec![]; N], |mut chunks, evals| {
    //     izip!(chunks.iter_mut(), evals).for_each(|(a, b)| a.push(b));
    //     chunks
    // });

    // let prover_openings = partial_openings
    //     .into_iter()
    //     .map(|evals| MultilinearPolynomial::from(evals).evaluate(&r_opening_remaining))
    //     .collect_vec();

    assert_eq!(prover_openings.len(), N);

    // Verification
    println!("\n/----------- Verification -----------/");

    let mut transcript = KeccakTranscript::new(&[]);
    assert_eq!(grand_product_output.len(), N);
    let output_mle = DensePolynomial::new_padded(grand_product_output.clone());
    let mut num_rounds = output_mle.get_num_vars();
    let mut r_grand_product: Vec<_> = transcript.challenge_vector(num_rounds);
    let mut grand_product_claim = output_mle.evaluate(&r_grand_product);

    for (i, layer_proof) in layer_proofs.iter().enumerate() {
        // layer sumcheck verification
        let (sumcheck_claim, r_sumcheck) = layer_proof
            .proof
            .verify(grand_product_claim, num_rounds, 3, &mut transcript)
            .unwrap();

        let eq_eval: F = r_grand_product
            .iter()
            .zip_eq(r_sumcheck.iter().rev())
            .map(|(&r_gp, &r_sc)| r_gp * r_sc + (F::ONE - r_gp) * (F::ONE - r_sc))
            .product();

        // cross-layer consistency check
        assert_eq!(
            layer_proof.left_claim * layer_proof.right_claim * eq_eval,
            sumcheck_claim
        );
        println!("layer {} - verified!", i + 1);

        let r_layer = transcript.challenge_scalar();
        grand_product_claim =
            layer_proof.left_claim + r_layer * (layer_proof.right_claim - layer_proof.left_claim);
        println!("layer {} - claim: {}", i + 1, grand_product_claim);

        r_grand_product = r_sumcheck.iter().rev().copied().collect();
        r_grand_product.push(r_layer); // pass updated r_grand_product to next layer
        num_rounds += 1;
    }

    // Verify openings

    // For a batch size of k, the first log2(k) elements of `r_grand_product`
    // form the point at which the output layer's MLE is evaluated. The remaining elements
    // then form the point at which the leaf layer's polynomials are evaluated.
    let (r_batch_index, r_opening) =
        r_grand_product.split_at(grand_product_output.len().next_power_of_two().log_2());

    assert_eq!(
        grand_product_output.len().next_power_of_two(),
        r_batch_index.len().pow2(),
    );

    // `r_batch_index` is used to combine the k claims in the batch into a single claim.
    let combined_output_claim: F = prover_openings
        .iter()
        .zip(EqPolynomial::evals(r_batch_index).iter())
        .map(|(hash, eq_eval)| *hash * eq_eval)
        .sum();

    assert_eq!(combined_output_claim, grand_product_claim);
    println!("fingerprint check - verified!");

    // Verifier recomputes openings to simulate PCS openings verification
    izip!(in_interleaved.chunks(chunk_size), prover_openings).for_each(
        |(coeffs, prover_opening)| {
            let verifier_opening =
                MultilinearPolynomial::from(coeffs.to_vec()).evaluate(&r_opening);
            assert_eq!(verifier_opening, prover_opening);
        },
    );
    println!("PCS.open(r_opening, commitments) == openings - verified!");
}

#[cfg(test)]
fn run_distributed_gkr_simulation<F: JoltField>(chunk_size: usize, N: usize, W: usize) {
    assert!(
        chunk_size.is_power_of_two(),
        "chunk_size must be a power of two"
    );
    assert!(W.is_power_of_two(), "num_workers must be a power of two");
    let chunk_size_worker = chunk_size / W;

    println!(
        "NUM_WORKERS={} | CHUNK_SIZE={}={}/worker | BATCH_SIZE={}",
        W, chunk_size, chunk_size_worker, N
    );

    let W_log2 = W.log_2();
    let leaves_len = chunk_size * N;
    let num_layers = (leaves_len / N).log_2();
    println!("num_layers: {}", num_layers);

    assert!(
        chunk_size_worker > 2,
        "chunk_size/worker must be greater than 2"
    );

    // Batch is as follows: [[1,1,1,...,1],[2,2,2,...,2],[3,3,3,...,3],...] each chunk has length `chunk_size`
    // So left/right polys are L(x): [[1,...,1],[2,...,2],[3,...,3],...]; R(x): [[1,...,1],[2,...,2],[3,...,3],...] each chunk has length `chunk_size/2`
    // The expected grand product results are thus [1^chunk_size, 2^chunk_size, 3^chunk_size, ...]
    let mut in_interleaved = vec![F::from(1); chunk_size];

    for i in 2..N + 1 {
        in_interleaved.extend(vec![F::from(i as u64); chunk_size]);
    }
    println!("in_interleaved: {:?}", in_interleaved);

    #[derive(Debug, Clone)]
    struct LayerCircuit<F: JoltField> {
        layer_idx: usize,
        polys: Vec<DenseInterleavedPolynomial<F>>,
    }

    struct LayerProof<F: JoltField> {
        proof: SumcheckInstanceProof<F, KeccakTranscript>,
        left_claim: F,
        right_claim: F,
    }

    println!("\n/---------- Construct layers ----------/");

    // Each worker gets chunks of size `chunk_size/W`, left/right polys are half of that.
    // THE ISSUE: if we combine worker chunks we'll have P_dgkr(X): [1..1, 2..2, 3..3, 1..1, 2..2, 3..3, ...]
    // But in local GKR, P_gkr(X) is instead: [1,1,1..1, 2,2,2..2, 3,3,3..3,...]; this breaks distributed sumcheck/GKR invariance.
    // Simple (naive) solutions:
    // 1. We can distribute chunks
    // - so worker chunks are P_w1: [1,1,1,...,1], P_w2: [2,2,2,...,2], P_w3: [3,3,3,...,3],...
    // - or P_w1: [[1,1,1,...,1],[2,2,2,...,2],..], P_w2: [[X,X,X,...,X],[Y,Y,Y,...,Y], ...],...
    // - this only works if each worker gets batch size which is *same* power of two, if `batch_size` is not power of two we cannot cleanly distribute
    // 2. We can add custom eq_simga(x, r)
    // - eq_simga(x, r)=eq(x, sigma(r, W)) implements custom wiring between worker layers and coordinor's final (output) layer
    // - since sigma(r) depends on num workers W, this will make verifier depend on how prover distributes work (bad)
    // - it also breaks openings checks
    //
    // New solution:
    // 3. We keep same eq for bind(eq) but implements custom wiring in `custom_eq_sumcheck_evals` (part of compute round message poly)
    // - Trade off is more complex orchestration when constructing and proving layers + extra communication between workers/coordinator.
    // - The upside is that it keeps same eq for for verifier, works with openings check, and allows any power of two number of workers.

    let mut w_interleaved = vec![vec![F::from(1); chunk_size_worker]; W];
    for i in 2..N + 1 {
        for w in 0..W {
            w_interleaved[w].extend(vec![F::from(i as u64); chunk_size_worker]);
        }
    }

    let input_layer = LayerCircuit {
        layer_idx: num_layers,
        polys: w_interleaved
            .iter()
            .cloned()
            .map(DenseInterleavedPolynomial::new)
            .collect(),
    };

    // println!(
    //     "input layer ({}) | polys: {:?}",
    //     input_layer.layer_idx,
    //     input_layer
    //         .polys
    //         .iter()
    //         .map(|p| &p.coeffs)
    //         .collect::<Vec<_>>()
    // );

    let worker_num_layers = num_layers - 2 - W_log2;
    let mut worker_layers = vec![input_layer];

    for i in 0..worker_num_layers {
        let prev_layer = &worker_layers[i];
        let layer_idx = num_layers - i - 1;

        let polys = prev_layer
            .polys
            .par_iter()
            .map(DenseInterleavedPolynomial::layer_output)
            .collect::<Vec<_>>();
        // println!(
        //     "worker layer {} | out: {:?}",
        //     layer_idx,
        //     polys.iter().map(|p| &p.coeffs).collect::<Vec<_>>()
        // );
        println!("-------------");

        // `MultilinearPolynomial::sumcheck_evals`'s the semantic pairing is (2i, 2i+1) in the coefficient array.
        // Left/right polys must contain at least 2 elements from each chunk,
        // otherwise `sumcheck_evals` would be pairing elements from unrelated chunks; breaking the distribution invariance.
        // TODO: if you comment this assert, and set W_log2:=1 verification passes for W>2; need to investigate this.
        assert!(
            polys[0].len() / 2 > N,
            "Left/right polys must contain at least 2 elements from each chunk"
        );

        let next_layer = LayerCircuit { layer_idx, polys };

        worker_layers.push(next_layer);
    }

    // Next levels are constructed/run by coordinator.
    // Alternatively, if we allow cross-worker communication, we can construct/run them by workers:
    // - At each subsequent layer, we can have workers send their partial results to smaller subnet (half the size)
    // - The proving is also done in this fashion.
    let coordinator_num_layers = W_log2 + 1;
    let mut coordinator_layers: Vec<LayerCircuit<F>> = Vec::with_capacity(coordinator_num_layers);

    let mut switched = false;
    for i in 0..coordinator_num_layers {
        let layer_idx = num_layers - worker_num_layers - i - 1;

        let next_layer_poly = if !switched {
            let prev_layer = worker_layers.last().unwrap();

            // What coordinator receives from workers (sub hashes)
            let prev_outputs_by_worker = prev_layer
                .polys
                .par_iter()
                .map(|poly| {
                    let (left, right) = poly.uninterleave();
                    izip!(&left, &right).map(|(a, b)| *a * *b).collect_vec()
                })
                .collect::<Vec<_>>();

            // println!(
            //     "worker layer {} | outputs by worker: {:?}",
            //     num_layers - worker_num_layers,
            //     prev_outputs_by_worker
            // );
            println!("-------------");

            // We restore order of results like this:
            // Worker results in our toy example: [[1, 4, 9, 16], [1, 4, 9, 16], ...]
            // The restored order is: [1,..1, 4...4, 9...9, 16...16] (interleave_n)
            let next_layer_coeffs = interleave_n(
                (0..W)
                    .into_iter()
                    .map(|w| prev_outputs_by_worker[w].chunks(1).collect_vec())
                    .collect::<Vec<_>>(),
            )
            .into_par_iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
            // println!(
            //     "worker layer {} | next_layer_coeffs: {:?}",
            //     num_layers - worker_num_layers,
            //     next_layer_coeffs
            // );
            switched = true;
            DenseInterleavedPolynomial::new(next_layer_coeffs)
        } else {
            // After switch run normally
            let prev_layer = coordinator_layers.last().unwrap();

            prev_layer.polys[0].layer_output()
        };
        println!("-------------");

        let next_layer = LayerCircuit {
            layer_idx,
            polys: vec![next_layer_poly],
        };

        // println!(
        //     "coordinator layer {} | poly {:?}",
        //     layer_idx, next_layer.polys[0].coeffs
        // );
        coordinator_layers.push(next_layer);
    }

    let grand_product_output = {
        let last_layer = coordinator_layers.last().unwrap();
        let (left, right) = last_layer.polys[0].uninterleave();
        izip!(left, right).map(|(a, b)| a * b).collect::<Vec<_>>()
    };

    // println!("gkr output {:?}", grand_product_output);

    println!("\n/---------- Coordinator prover ----------/");

    //------ Output layer (N) prover
    let mut layer_proofs = vec![];
    let mut transcript = KeccakTranscript::new(&[]);

    let output_mle = DensePolynomial::new_padded(grand_product_output.clone());
    let mut num_rounds = output_mle.get_num_vars();
    let mut r_grand_product: Vec<_> = transcript.challenge_vector(num_rounds);
    // gkr output claim, will be updated as output claim of each subsequent layer as we progress to input layer
    let mut grand_product_claim = output_mle.evaluate(&r_grand_product);

    // GKR proves from output layer to input layer
    for mut layer in coordinator_layers.into_iter().rev() {
        println!("layer 1 rounds: {} -------", num_rounds);

        let mut eq_poly = SplitEqPolynomial::new(&r_grand_product);

        let (proof, r_sumcheck, (left_claim, right_claim)) =
            layer.polys[0].prove_sumcheck(&grand_product_claim, &mut eq_poly, &mut transcript);

        layer_proofs.push(LayerProof {
            proof,
            left_claim,
            right_claim,
        });

        let r_layer = transcript.challenge_scalar();
        grand_product_claim = left_claim + r_layer * (right_claim - left_claim);

        println!("layer {} claim: {:?}", layer.layer_idx, grand_product_claim);

        r_grand_product = r_sumcheck.iter().rev().copied().collect();
        r_grand_product.push(r_layer); // pass r_grand_product to next layer
        num_rounds += 1;
    }

    //------ Distributed layers prover

    println!("\n/------------ Worker prover ------------/");

    let mut chunk_size_per_worker = 2;
    for mut layer in worker_layers.iter().cloned().rev() {
        // println!(
        //     "layer {} rounds: {:?} polys: {:?}",
        //     layer.layer_idx,
        //     num_rounds,
        //     layer.polys.iter().map(|p| &p.coeffs).collect::<Vec<_>>()
        // );

        let mut eq_poly = SplitEqPolynomial::new(&r_grand_product);

        // // Setting eq_poly.E1_len to 1 turns it into oridnary EqPolynomial
        // eq_poly.E1_len = 1;
        // eq_poly.E1 = vec![F::one()];
        // eq_poly.E2 = EqPolynomial::evals(&r_grand_product);
        // eq_poly.E2_len = eq_poly.E2.len();

        let (proof, r_sumcheck, (left_claim, right_claim)) =
            SumcheckInstanceProof::<F, KeccakTranscript>::simulate_prove_cubic_distributed_chunk_wize(
                &grand_product_claim,
                num_rounds,
                &mut eq_poly,
                &mut layer.polys,
                chunk_size_per_worker,
                &mut transcript,
            );

        layer_proofs.push(LayerProof {
            proof,
            left_claim,
            right_claim,
        });

        let r_layer = transcript.challenge_scalar();
        grand_product_claim = left_claim + r_layer * (right_claim - left_claim);

        println!("layer {} claim: {:?}", layer.layer_idx, grand_product_claim);

        r_grand_product = r_sumcheck.iter().rev().copied().collect();
        r_grand_product.push(r_layer); // pass r_grand_product2 to next layer
        num_rounds += 1;
        chunk_size_per_worker *= 2;
        println!("-------------");
    }

    // Compute openings
    println!("\n/---------- Compute openings ----------/");

    let (_, r_opening) =
        r_grand_product.split_at(grand_product_output.len().next_power_of_two().log_2());
    let (r_opening_worker, r_opening_remaining) =
        r_opening.split_at(chunk_size_worker.next_power_of_two().log_2());
    println!(
        "r_grand_product {:?} r_opening {:?} r_opening_worker {:?} r_opening_remaining {:?}",
        r_grand_product.len(),
        r_opening.len(),
        r_opening_worker.len(),
        r_opening_remaining.len()
    );

    let partial_openings: Vec<_> = worker_layers[0]
        .polys
        .iter()
        .map(|poly| {
            let w_opennings = poly
                .coeffs
                .chunks(chunk_size_worker)
                .map(|w_chunk| {
                    MultilinearPolynomial::from(w_chunk.to_vec()).evaluate(&r_opening_worker)
                })
                .collect_vec();
            assert_eq!(w_opennings.len(), N);
            w_opennings
        })
        .fold(vec![vec![]; N], |mut chunks, evals| {
            izip!(chunks.iter_mut(), evals).for_each(|(a, b)| a.push(b));
            chunks
        });

    let prover_openings = partial_openings
        .into_iter()
        .map(|evals| MultilinearPolynomial::from(evals).evaluate(&r_opening_remaining))
        .collect_vec();

    assert_eq!(prover_openings.len(), N);

    // Verification
    println!("\n/----------- Verification -----------/");

    let mut transcript = KeccakTranscript::new(&[]);
    assert_eq!(grand_product_output.len(), N);
    let output_mle = DensePolynomial::new_padded(grand_product_output.clone());
    let mut num_rounds = output_mle.get_num_vars();
    let mut r_grand_product: Vec<_> = transcript.challenge_vector(num_rounds);
    let mut grand_product_claim = output_mle.evaluate(&r_grand_product);

    for (i, layer_proof) in layer_proofs.iter().enumerate() {
        // layer sumcheck verification
        let (sumcheck_claim, r_sumcheck) = layer_proof
            .proof
            .verify(grand_product_claim, num_rounds, 3, &mut transcript)
            .unwrap();

        let eq_eval: F = r_grand_product
            .iter()
            .zip_eq(r_sumcheck.iter().rev())
            .map(|(&r_gp, &r_sc)| r_gp * r_sc + (F::ONE - r_gp) * (F::ONE - r_sc))
            .product();

        // cross-layer consistency check
        assert_eq!(
            layer_proof.left_claim * layer_proof.right_claim * eq_eval,
            sumcheck_claim
        );
        println!("layer {} - verified!", i + 1);

        let r_layer = transcript.challenge_scalar();
        grand_product_claim =
            layer_proof.left_claim + r_layer * (layer_proof.right_claim - layer_proof.left_claim);
        println!("layer {} - claim: {}", i + 1, grand_product_claim);

        r_grand_product = r_sumcheck.iter().rev().copied().collect();
        r_grand_product.push(r_layer); // pass updated r_grand_product to next layer
        num_rounds += 1;
    }

    // Verify openings

    // For a batch size of k, the first log2(k) elements of `r_grand_product`
    // form the point at which the output layer's MLE is evaluated. The remaining elements
    // then form the point at which the leaf layer's polynomials are evaluated.
    let (r_batch_index, r_opening) =
        r_grand_product.split_at(grand_product_output.len().next_power_of_two().log_2());

    assert_eq!(
        grand_product_output.len().next_power_of_two(),
        r_batch_index.len().pow2(),
    );

    // `r_batch_index` is used to combine the k claims in the batch into a single claim.
    let combined_output_claim: F = prover_openings
        .iter()
        .zip(EqPolynomial::evals(r_batch_index).iter())
        .map(|(hash, eq_eval)| *hash * eq_eval)
        .sum();

    assert_eq!(combined_output_claim, grand_product_claim);
    println!("fingerprint check - verified!");

    // Verifier recomputes openings to simulate PCS openings verification
    izip!(in_interleaved.chunks(chunk_size), prover_openings).for_each(
        |(coeffs, prover_opening)| {
            let verifier_opening =
                MultilinearPolynomial::from(coeffs.to_vec()).evaluate(&r_opening);
            assert_eq!(verifier_opening, prover_opening);
        },
    );
    println!("PCS.open(r_opening, commitments) == openings - verified!");
}

#[test]
fn test_local_gkr_simulation() {
    type F = ark_bn254::Fr;

    let chunk_size: usize = env::var("CHUNK_SIZE")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap();
    let N: usize = env::var("BATCH_SIZE")
        .unwrap_or_else(|_| "4".to_string())
        .parse()
        .unwrap();

    println!("CHUNK_SIZE={}; BATCH_SIZE={}", chunk_size, N);

    let leaves_len = chunk_size * N;
    let num_layers = (leaves_len / N).log_2();
    println!("num_layers: {}", num_layers);

    let mut in_interleaved = vec![1; chunk_size]
        .into_iter()
        .map(F::from)
        .collect::<Vec<_>>();

    for i in 2..N + 1 {
        in_interleaved.extend(vec![F::from(i as u64); chunk_size]);
    }
    println!("in_interleaved: {:?}", in_interleaved);

    struct LayerCircuit<F: JoltField> {
        layer_idx: usize,
        poly: DenseInterleavedPolynomial<F>,
    }

    struct LayerProof<F: JoltField> {
        proof: SumcheckInstanceProof<F, KeccakTranscript>,
        left_claim: F,
        right_claim: F,
    }

    println!("\n/----------- Construct layers -----------/");

    let input_layer = LayerCircuit {
        layer_idx: num_layers,
        poly: DenseInterleavedPolynomial::new(in_interleaved.clone()),
    };

    let mut layers = vec![input_layer];

    for i in 0..num_layers - 1 {
        let prev_layer = &layers[i];
        let next_layer_poly = prev_layer.poly.layer_output();
        let layer_idx = num_layers - i - 1;

        println!("layer {} | poly {:?}", layer_idx, next_layer_poly.coeffs);

        let next_layer = LayerCircuit {
            layer_idx,
            poly: next_layer_poly,
        };

        layers.push(next_layer);
    }

    let last_layer = layers.last().unwrap();
    let (last_left, last_right) = last_layer.poly.uninterleave();
    println!("last_left: {:?} last_right: {:?}", last_left, last_right);
    let grand_product_output = izip!(last_left, last_right)
        .map(|(left, right)| left * right)
        .collect::<Vec<_>>();

    println!("gkr output {:?}", grand_product_output);

    println!("\n/----------- Local prover -----------/");

    //------ Output layer (N) prover
    let mut layer_proofs = vec![];
    let mut transcript = KeccakTranscript::new(&[]);

    let output_mle = DensePolynomial::new_padded(grand_product_output.clone());
    let mut num_rounds = output_mle.get_num_vars();
    let mut r_grand_product: Vec<_> = transcript.challenge_vector(num_rounds);
    let mut grand_product_claim = output_mle.evaluate(&r_grand_product);

    for mut layer in layers.into_iter().rev() {
        println!(
            "----------layer {} rounds: {:?}----------",
            layer.layer_idx, num_rounds
        );
        println!("poly: {:?}", layer.poly.len());

        let mut eq_poly = SplitEqPolynomial::new(&r_grand_product);

        let (proof, r_sumcheck, final_evals) =
            layer
                .poly
                .prove_sumcheck(&grand_product_claim, &mut eq_poly, &mut transcript);

        let (left_claim, right_claim) = final_evals;

        layer_proofs.push(LayerProof {
            proof,
            left_claim,
            right_claim,
        });

        let r_layer = transcript.challenge_scalar();
        grand_product_claim = left_claim + r_layer * (right_claim - left_claim);

        println!("layer {} claim: {:?}", layer.layer_idx, grand_product_claim);

        r_grand_product = r_sumcheck.iter().rev().copied().collect();
        r_grand_product.push(r_layer); // pass r_grand_product to next layer
        num_rounds += 1;
    }

    // Compute openings
    let (_, r_opening) =
        r_grand_product.split_at(grand_product_output.len().next_power_of_two().log_2());

    let prover_openings = in_interleaved
        .chunks(chunk_size)
        .map(|coeffs| MultilinearPolynomial::from(coeffs.to_vec()).evaluate(&r_opening))
        .collect::<Vec<_>>();

    // Verification
    println!("\n/----------- Verification -----------/");

    let mut transcript = KeccakTranscript::new(&[]);
    assert_eq!(grand_product_output.len(), N);
    let output_mle = DensePolynomial::new_padded(grand_product_output.clone());
    let mut num_rounds = output_mle.get_num_vars();
    let mut r_grand_product: Vec<_> = transcript.challenge_vector(num_rounds);
    let mut grand_product_claim = output_mle.evaluate(&r_grand_product);

    for (i, layer_proof) in layer_proofs.iter().enumerate() {
        let (sumcheck_claim, r_sumcheck) = layer_proof
            .proof
            .verify(grand_product_claim, num_rounds, 3, &mut transcript)
            .unwrap();

        let eq_eval: F = r_grand_product
            .iter()
            .zip_eq(r_sumcheck.iter().rev())
            .map(|(&r_gp, &r_sc)| r_gp * r_sc + (F::ONE - r_gp) * (F::ONE - r_sc))
            .product();

        // cross-layer consistency check
        assert_eq!(
            layer_proof.left_claim * layer_proof.right_claim * eq_eval,
            sumcheck_claim
        );
        println!("layer {} - verified!", i + 1);

        let r_layer = transcript.challenge_scalar();
        grand_product_claim =
            layer_proof.left_claim + r_layer * (layer_proof.right_claim - layer_proof.left_claim);
        println!("layer {} - claim: {}", i + 1, grand_product_claim);

        r_grand_product = r_sumcheck.iter().rev().copied().collect();
        r_grand_product.push(r_layer); // pass r_grand_product2 to next layer
        num_rounds += 1;
    }

    // Verify openings

    // For a batch size of k, the first log2(k) elements of `r_grand_product`
    // form the point at which the output layer's MLE is evaluated. The remaining elements
    // then form the point at which the leaf layer's polynomials are evaluated.
    let (r_batch_index, r_opening) =
        r_grand_product.split_at(grand_product_output.len().next_power_of_two().log_2());

    assert_eq!(
        grand_product_output.len().next_power_of_two(),
        r_batch_index.len().pow2(),
    );

    // `r_batch_index` is used to combine the k claims in the batch into a single claim.
    let combined_output_claim: F = prover_openings
        .iter()
        .zip(EqPolynomial::evals(r_batch_index).iter())
        .map(|(hash, eq_eval)| *hash * eq_eval)
        .sum();

    assert_eq!(combined_output_claim, grand_product_claim);

    // Verifier recomputes openings to simulate PCS openings verification
    izip!(in_interleaved.chunks(chunk_size), prover_openings).for_each(
        |(coeffs, prover_opening)| {
            let verifier_opening =
                MultilinearPolynomial::from(coeffs.to_vec()).evaluate(&r_opening);
            assert_eq!(verifier_opening, prover_opening);
        },
    );
}

#[test]
fn test_distributed_gkr_simulation() {
    let chunk_size: usize = env::var("CHUNK_SIZE")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap();
    let N: usize = env::var("BATCH_SIZE")
        .unwrap_or_else(|_| "4".to_string())
        .parse()
        .unwrap();
    let W = env::var("NUM_WORKERS")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .unwrap();

    run_simulation_dbgp_batch_wize::<ark_bn254::Fr>(chunk_size, N, W);
}

#[test]
fn test_cases_distributed_gkr_simulation() {
    let W = env::var("NUM_WORKERS")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .unwrap();

    let min_chunk_size = W * 4;

    let cases = [
        (min_chunk_size, 4),
        (min_chunk_size, min_chunk_size),
        (min_chunk_size, min_chunk_size * 2),
        (min_chunk_size, 3),
        // (1 << 9, min_chunk_size), // read_write mini
        (1 << 13, 102), // read_write
        (1 << 16, 75),  // init_final
    ];
    for (chunk_size, N) in cases {
        println!("/------------ Test case start ------------/");
        run_simulation_dbgp_batch_wize::<ark_bn254::Fr>(chunk_size, N, W);
        println!("/-----------------------------------------/");
    }
}

/// Compute sumcheck evaluations for the equality polynomial in a distributed setting
/// with `W >= 2` workers (assume `W` is a power of two), using the same logical wiring
/// as the local, non-distributed sumcheck.
///
/// Conceptually, we do not reshuffle coefficient arrays. Instead, we view the global
/// index space at this round (`0..EQ_HALF`) as split into contiguous blocks, and we
/// stripe these blocks across workers round-robin. Each worker’s local index walks
/// through its own striped view. This reproduces the exact pairs `(2i, 2i+1)` the local
/// prover would multiply, but without materializing any intermediate vectors.
///
/// Mapping without allocation:
/// - `BLOCK = global_eq_pairs / W` (must divide evenly)
/// - For a worker `w` and local index `i`, let `k = i / BLOCK` and `off = i % BLOCK`.
/// - The corresponding global index is `g = (k*W + w)*BLOCK + off`.
///
/// We then perform the standard LowToHigh sumcheck evaluation at `g`:
/// `evals[j] = P(2g + j)` for `j` on the univariate degree points.
fn custom_eq_sumcheck_evals<F: JoltField>(
    eq_evals: &[F],
    index: usize,
    degree: usize,
    global_eq_pairs: usize,
    worker: usize,
    num_workers: usize,
) -> Vec<F> {
    // println!("custom eq | chunk_mle: {} index {}", global_eq_pairs, index);
    debug_assert!(num_workers >= 2 && num_workers.is_power_of_two());
    debug_assert!(global_eq_pairs >= num_workers);
    debug_assert_eq!(global_eq_pairs % num_workers, 0);

    // Compute the global index without allocating intermediate vectors.
    let block_size = global_eq_pairs / num_workers;
    let k = index / block_size; // which block within the worker's sequence
    let offset = index % block_size; // position inside that block
    let global_block = k * num_workers + worker;
    let global_index = global_block * block_size + offset;

    let mut evals = vec![F::zero(); degree];
    evals[0] = eq_evals[2 * global_index];
    if degree == 1 {
        return evals;
    }
    let mut eval = eq_evals[2 * global_index + 1];
    let m = eval - evals[0];
    for i in 1..degree {
        eval += m;
        evals[i] = eval;
    }
    evals
}
// /// Return a worker-local permutation of `(E1, E2)` consistent with the
// /// `custom_eq_permute` order on flattened EQ evals, preserving the E1/E2 factorization.
// ///
// /// Feasibility and behavior:
// /// - If `E1_len == 1` (first half fully bound), the EQ table reduces to `E2` and the
// ///   permutation is identical to `custom_eq_permute` applied to `E2`. We return
// ///   `(E1, E2_worker)` with `E1` unchanged and `E2_worker` containing only the rows
// ///   assigned to `worker` in the permuted order.
// /// - If `E1_len > 1`, a consistent factorization exists if and only if the worker block
// ///   size in pair units, `block_pairs = global_eq_pairs / W`, is a multiple of
// ///   `pairs_per_row = E1_len / 2`. In that case, the permutation separates as a pure
// ///   row permutation on `E2` with `E1` unchanged; we return `(E1, E2_worker)` where
// ///   `E2_worker` lists the selected rows for `worker` in order. Otherwise, the
// ///   permutation is not separable across `(E1, E2)`, and this function will panic.
// fn split_eq_poly_permute<F: JoltField>(
//     eq_poly: &SplitEqPolynomial<F>,
//     global_eq_pairs: usize,
//     worker: usize,
//     num_workers: usize,
// ) -> (Vec<F>, Vec<F>) {
//     debug_assert!(num_workers >= 2 && num_workers.is_power_of_two());
//     debug_assert!(global_eq_pairs >= num_workers);
//     debug_assert_eq!(global_eq_pairs % num_workers, 0);

//     let e1_len = eq_poly.E1_len;
//     let e2_len = eq_poly.E2_len;

//     if e1_len == 1 {
//         // Degenerate case: E1 is fully bound; the factorization is trivial.
//         let e1_out = eq_poly.E1[..e1_len].to_vec();
//         let e2_out = custom_eq_permute(&eq_poly.E2[..e2_len], global_eq_pairs, worker, num_workers);
//         return (e1_out, e2_out);
//     }

//     // General case: attempt to separate the permutation across E2 rows and E1 columns.
//     assert_eq!(e1_len % 2, 0, "E1_len must be even");
//     let pairs_per_row = e1_len / 2; // number of (low, high) pairs per E2 row
//     let num_pairs_total = e2_len * pairs_per_row;
//     assert!(
//         num_pairs_total % global_eq_pairs == 0,
//         "num_pairs_total must be divisible by global_eq_pairs"
//     );

//     let eq_dummy = (0u64..((e1_len * e2_len) as u64))
//         .map(F::from)
//         .collect_vec();
//     println!(
//         "eq_dummy_permuted_16: {:?}",
//         [
//             custom_eq_permute(&eq_dummy, global_eq_pairs, 0, num_workers),
//             custom_eq_permute(&eq_dummy, global_eq_pairs, 1, num_workers)
//         ]
//     );
//     let mut dummy_split_eq = SplitEqPolynomial::new(&(0u64..4).map(F::from).collect_vec());
//     println!(
//         "dummy_split_eq: [{:?}, {:?}]",
//         dummy_split_eq.E1, dummy_split_eq.E2
//     );

//     println!("dummy_split_eq merged: {:?}", dummy_split_eq.merge().Z);
//     for w in 0..num_workers {
//         let dummy_split_eq_merged_permuted = custom_eq_permute(
//             &dummy_split_eq.merge().Z,
//             global_eq_pairs,
//             worker,
//             num_workers,
//         );
//         dummy_split_eq.E1 = custom_eq_permute(&dummy_split_eq.E1, global_eq_pairs, w, num_workers);
//         dummy_split_eq.E2 = custom_eq_permute(&dummy_split_eq.E2, global_eq_pairs, w, num_workers);
//         dummy_split_eq.E1_len /= 2;
//         dummy_split_eq.E2_len /= 2;
//         assert_eq!(
//             dummy_split_eq.merge().Z,
//             dummy_split_eq_merged_permuted[w * 4..w * 4 + 4]
//         );
//     }
//     let block_pairs = global_eq_pairs / num_workers;
//     println!(
//         "e1_len {} e2_len {} block_pairs {} pairs_per_row {}",
//         e1_len, e2_len, block_pairs, pairs_per_row
//     );
//     assert!(
//         block_pairs % pairs_per_row == 0,
//         "Permutation not separable: block_pairs (global_eq_pairs/num_workers) must be a multiple of pairs_per_row (E1_len/2)."
//     );
//     let rows_per_block = block_pairs / pairs_per_row;
//     assert!(
//         e2_len % (rows_per_block * num_workers) == 0,
//         "E2_len must be divisible by rows_per_block * num_workers"
//     );

//     let cols = e2_len / (rows_per_block * num_workers);

//     let e1_out = eq_poly.E1[..e1_len].to_vec(); // unchanged
//     let mut e2_out = Vec::with_capacity(e2_len / num_workers);
//     for k in 0..cols {
//         let base_row = (k * num_workers + worker) * rows_per_block;
//         for r in 0..rows_per_block {
//             e2_out.push(eq_poly.E2[base_row + r]);
//         }
//     }

//     debug_assert_eq!(e2_out.len(), e2_len / num_workers);
//     (e1_out, e2_out)
// }

/// Interleave N vectors:
/// [v0[0], v1[0], v2[0], ..., v0[1], v1[1], v2[1], ...]
/// Works for ragged inputs (shorter inner vecs are just skipped).
pub fn interleave_n<T>(v: Vec<impl IntoIterator<Item = T>>) -> Vec<T> {
    if v.is_empty() {
        return Vec::new();
    }

    // Turn each inner Vec<T> into an iterator so we can move out of it.
    let mut iters: Vec<_> = v.into_iter().map(|inner| inner.into_iter()).collect();

    let total_len: usize = iters.iter().map(|it| it.size_hint().0).sum();
    let mut out = Vec::with_capacity(total_len);

    loop {
        let mut progressed = false;

        for it in iters.iter_mut() {
            if let Some(x) = it.next() {
                out.push(x);
                progressed = true;
            }
        }

        if !progressed {
            break;
        }
    }

    out
}

fn uninterleave_with_padding<F: JoltField>(v: &[F]) -> (Vec<F>, Vec<F>) {
    let n = v.len() / 2;
    let n_padded = n.next_power_of_two();
    (
        v.iter()
            .copied()
            .step_by(2)
            .pad_using(n_padded, |_| F::ZERO)
            .collect(),
        v.iter()
            .copied()
            .skip(1)
            .step_by(2)
            .pad_using(n_padded, |_| F::ZERO)
            .collect(),
    )
}

fn uninterleave<T: Clone + Send + Sync>(v: &[T]) -> (Vec<T>, Vec<T>) {
    (
        v.par_iter().cloned().step_by(2).collect(),
        v.par_iter().cloned().skip(1).step_by(2).collect(),
    )
}

fn sigma<F: JoltField>(v: Vec<F>) -> Vec<F> {
    let n = v.len();
    assert!(n.is_power_of_two());

    let mut out = Vec::with_capacity(n);
    out.extend(v.iter().step_by(2)); // evens
    out.extend(v.iter().skip(1).step_by(2)); // odds
    out
}

// fn sigma_r_split<F: JoltField>(r: &[F]) -> Vec<F> {
//     let n = r.len();
//     let mut r_sigma = Vec::with_capacity(n);
//     r_sigma.push(r[n - 1]);
//     r_sigma.extend_from_slice(&r[..n - 1]);
//     r_sigma
// }

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct SumcheckInstanceProof<F: JoltField, ProofTranscript: Transcript> {
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
    pub fn new(
        compressed_polys: Vec<CompressedUniPoly<F>>,
    ) -> SumcheckInstanceProof<F, ProofTranscript> {
        SumcheckInstanceProof {
            compressed_polys,
            _marker: PhantomData,
        }
    }

    /// Verify this sumcheck proof.
    /// Note: Verification does not execute the final check of sumcheck protocol: g_v(r_v) = oracle_g(r),
    /// as the oracle is not passed in. Expected that the caller will implement.
    ///
    /// Params
    /// - `claim`: Claimed evaluation
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F>), ProofVerifyError> {
        let mut e = claim;
        let mut r: Vec<F> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            // verify degree bound
            if self.compressed_polys[i].degree() != degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    self.compressed_polys[i].degree(),
                ));
            }

            // append the prover's message to the transcript
            self.compressed_polys[i].append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i = transcript.challenge_scalar();
            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i using the hint
            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
        }

        Ok((e, r))
    }
}

/// Helper function to encapsulate the common subroutine for sumcheck with eq poly factor:
/// - Compute the linear factor E_i(X) from the current eq-poly
/// - Reconstruct the cubic polynomial s_i(X) = E_i(X) * t_i(X) for the i-th round
/// - Compress the cubic polynomial
/// - Append the compressed polynomial to the transcript
/// - Derive the challenge for the next round
/// - Bind the cubic polynomial to the challenge
/// - Update the claim as the evaluation of the cubic polynomial at the challenge
///
/// Returns the derived challenge
#[inline]
pub fn process_eq_sumcheck_round<F: JoltField, ProofTranscript: Transcript>(
    quadratic_evals: (F, F), // (t_i(0), t_i(infty))
    eq_poly: &mut GruenSplitEqPolynomial<F>,
    polys: &mut Vec<CompressedUniPoly<F>>,
    r: &mut Vec<F>,
    claim: &mut F,
    transcript: &mut ProofTranscript,
) -> F {
    let scalar_times_w_i = eq_poly.current_scalar * eq_poly.w[eq_poly.current_index - 1];

    let cubic_poly = UniPoly::from_linear_times_quadratic_with_hint(
        // The coefficients of `eq(w[(n - i)..], r[..i]) * eq(w[n - i - 1], X)`
        [
            eq_poly.current_scalar - scalar_times_w_i,
            scalar_times_w_i + scalar_times_w_i - eq_poly.current_scalar,
        ],
        quadratic_evals.0,
        quadratic_evals.1,
        *claim,
    );

    // Compress and add to transcript
    let compressed_poly = cubic_poly.compress();
    compressed_poly.append_to_transcript(transcript);

    // Derive challenge
    let r_i: F = transcript.challenge_scalar();
    r.push(r_i);
    polys.push(compressed_poly);

    // Evaluate for next round's claim
    *claim = cubic_poly.evaluate(&r_i);

    // Bind eq_poly for next round
    eq_poly.bind(r_i);

    r_i
}
