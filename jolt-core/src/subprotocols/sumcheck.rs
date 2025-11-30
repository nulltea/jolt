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
#[cfg(test)]
use crate::poly::sparse_interleaved_poly::SparseInterleavedPolynomial;
use crate::poly::spartan_interleaved_poly::SpartanInterleavedPolynomial;
use crate::poly::split_eq_poly::DistributedSplitEqPolynomial;
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
use ark_std::rand::Rng;
use ark_std::test_rng;
use itertools::{chain, concat, interleave, izip, Itertools};
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::{env, panic};
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
                "!round: {} eq_poly: E1_len {} E2_len {}",
                _round, eq_poly.E1_len, eq_poly.E2_len
            );

            let cubic_poly = self.compute_cubic(eq_poly, previous_claim);
            let compressed_poly = cubic_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            // derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar();
            println!("r_j: {:?}", r_j);

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
    pub fn simulate_prove_cubic_distributed_batch_wize(
        claim: &F,
        num_rounds: usize,
        eq_polys: &mut [DistributedSplitEqPolynomial<F>],
        r_grand_product: &[F],
        workers_polys: &mut [DenseInterleavedPolynomial<F>],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, (F, F)) {
        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        let num_workers = workers_polys.len();
        let worker_rounds = eq_polys[0].get_num_vars() - 2;

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
                    println!("worker {worker} poly: {:?}", &poly.len());
                    println!(
                        "worker {worker} eq_poly E1_len: {} E2_len: {}",
                        &eq_polys[worker].E1_len, &eq_polys[worker].E2_len
                    );
                    let evals = dense_interleaved_sumcheck_evals(poly, &eq_polys[worker]);
                    println!("----------------");
                    evals
                })
                .reduce(|mut eval_points, eval_points_next| {
                    izip!(eval_points.iter_mut(), eval_points_next).for_each(|(a, b)| *a += b);
                    eval_points
                })
                .unwrap();

            // println!("-------");
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

        // let eq_poly_remaining_evals: Vec<Vec<F>> = eq_polys
        //     .iter()
        //     .map(|poly| {
        //         println!("E1_len {} E2_len {}", poly.E1_len, poly.E2_len);
        //         poly.E2[..poly.E2_len].to_vec()
        //     })
        //     .collect();

        let mut poly = DenseInterleavedPolynomial::new(
            (0..num_workers)
                .flat_map(|w| poly_remaining_evals[w].clone())
                .collect_vec(),
        );
        // let mut eq_poly = SplitEqPolynomial::new_bound(
        //     vec![F::ZERO],
        //     (0..num_workers)
        //         .flat_map(|w| eq_poly_remaining_evals[w].clone())
        //         .collect_vec(),
        //     remaining_rounds,
        // );

        let mut eq_poly = {
            let mut eq_poly = SplitEqPolynomial::new(r_grand_product);
            r.iter().for_each(|r| eq_poly.bind(*r));
            eq_poly
        };

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

    #[cfg(test)]
    pub fn simulate_prove_cubic_distributed_sparse_batch_wize(
        claim: &F,
        num_rounds: usize,
        eq_polys: &mut [DistributedSplitEqPolynomial<F>],
        r_grand_product: &[F],
        workers_polys: &mut [SparseInterleavedPolynomial<F>],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, (F, F)) {
        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        let num_workers = workers_polys.len();
        let worker_rounds = eq_polys[0].get_num_vars() - 2;

        let remaining_rounds = num_rounds - worker_rounds;
        println!(
            "num_rounds: {} worker_rounds {} remaining_rounds {}",
            num_rounds, worker_rounds, remaining_rounds
        );

        println!(
            "poly: {} eq_poly: [{}]",
            workers_polys[0].dense_len,
            eq_polys
                .iter()
                .map(|eq| format!("E1_len={} E2_len={}", eq.E1_len, eq.E2_len))
                .collect::<Vec<String>>()
                .join(", ")
        );

        for _round in 0..worker_rounds {
            // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
            // for points {0, ..., |g(x)|}
            // println!("worker round {} poly: {}", _round, workers_polys[0].len());

            // let mle_half = workers_polys[0].len() / 2;

            println!(
                "round {} polys {:?} eq_poly: [{}]",
                _round,
                workers_polys.iter().map(|p| p.dense_len).collect_vec(),
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
                    println!("worker {worker} poly: {:?}", &poly.dense_len);
                    println!(
                        "worker {worker} eq_poly E1_len: {} E2_len: {}",
                        &eq_polys[worker].E1_len, &eq_polys[worker].E2_len
                    );
                    println!(
                        "worker {worker} poly is coallesed {:?}",
                        &poly.coalesced.is_some()
                    );
                    let evals = sparse_interleaved_sumcheck_evals(poly, &eq_polys[worker], worker);
                    println!("----------------");
                    evals
                })
                .reduce(|mut eval_points, eval_points_next| {
                    izip!(eval_points.iter_mut(), eval_points_next).for_each(|(a, b)| *a += b);
                    eval_points
                })
                .unwrap();

            // println!("-------");
            // println!("eval points: {:?}", eval_points);
            // println!("--------------");

            eval_points.insert(1, previous_claim - eval_points[0]);
            let univariate_poly = UniPoly::from_evals(&eval_points);
            let compressed_poly = univariate_poly.compress();
            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            println!("r_j: {:?}", r_j);
            r.push(r_j);

            // bound all tables to the verifier's challenge
            workers_polys.par_iter_mut().for_each(|poly| poly.bind(r_j));
            eq_polys.par_iter_mut().for_each(|poly| poly.bind(r_j));
            previous_claim = univariate_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        println!(
            "poly: {} eq_poly: [{}]",
            workers_polys[0].dense_len,
            eq_polys
                .iter()
                .map(|eq| format!("E1_len={} E2_len={}", eq.E1_len, eq.E2_len))
                .collect::<Vec<String>>()
                .join(", ")
        );

        let poly_remaining_evals: Vec<Vec<F>> = workers_polys
            .iter()
            .map(|spoly| spoly.coalesced.as_ref().unwrap())
            .map(|poly| poly.coeffs[..poly.len()].to_vec())
            .collect();

        // let eq_poly_remaining_evals: Vec<Vec<F>> = eq_polys
        //     .iter()
        //     .map(|poly| {
        //         println!("E1_len {} E2_len {}", poly.E1_len, poly.E2_len);
        //         poly.E2[..poly.E2_len].to_vec()
        //     })
        //     .collect();

        let mut poly = DenseInterleavedPolynomial::new(
            (0..num_workers)
                .flat_map(|w| poly_remaining_evals[w].clone())
                .collect_vec(),
        );
        // let mut eq_poly = SplitEqPolynomial::new_bound(
        //     vec![F::ZERO],
        //     (0..num_workers)
        //         .flat_map(|w| eq_poly_remaining_evals[w].clone())
        //         .collect_vec(),
        //     remaining_rounds,
        // );

        let mut eq_poly = {
            let mut eq_poly = SplitEqPolynomial::new(r_grand_product);
            r.iter().for_each(|r| eq_poly.bind(*r));
            eq_poly
        };

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

/// Evaluate the cubic sumcheck round polynomial for a `DenseInterleavedPolynomial`
/// against a (possibly distributed) SplitEqPolynomial, using Dao–Thaler.
///
/// - `poly` is interleaved as [L0, R0, L1, R1, ...].
/// - `eq_poly` is either:
///     * factored as E2 * E1 (E1_len > 1, Dao–Thaler mode), or
///     * collapsed into E2 only (E1_len == 1, linear-time mode).
#[cfg(test)]
fn dense_interleaved_sumcheck_evals<F: JoltField>(
    poly: &DenseInterleavedPolynomial<F>,
    eq_poly: &DistributedSplitEqPolynomial<F>,
) -> Vec<F> {
    println!("dense_interleaved_sumcheck_evals");
    // We use the Dao–Thaler optimization for the EQ polynomial, so there are two cases:
    //   1) E1_len == 1: fully bound inner dimension → standard linear-time sumcheck.
    //   2) E1_len > 1:  factored Eq = E2(i_A,i_B) * E1(i_C) → nested summation.
    let cubic_evals = if eq_poly.E1_len == 1 {
        // ---------------- linear-time mode: no Dao–Thaler factorization left ----------------
        //
        // At this point, E2 already contains the full Eq evaluations over the remaining
        // variables, aligned 1:1 with the points that `poly` represents. We simply
        // zip chunks of P with chunks of Eq and do the standard cubic evaluation.
        poly.par_chunks(4)
            .zip(eq_poly.E2.par_chunks(2))
            .map(|(layer_chunk, eq_chunk)| {
                // Compute Eq(r) at three relevant points j ∈ {0, 2, 3} from two stored
                // values using the standard 4-point interpolation trick.
                let eq_evals = {
                    let eval_point_0 = eq_chunk[0];
                    let m_eq = eq_chunk[1] - eq_chunk[0];
                    let eval_point_2 = eq_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                };

                // Interleaved [L0, R0, L1, R1] chunk for this point.
                let left = (
                    *layer_chunk.first().unwrap_or(&F::zero()),
                    *layer_chunk.get(2).unwrap_or(&F::zero()),
                );
                let right = (
                    *layer_chunk.get(1).unwrap_or(&F::zero()),
                    *layer_chunk.get(3).unwrap_or(&F::zero()),
                );

                // Evaluate left(r) and right(r) at j = 2, 3 using affine interpolation.
                let m_left = left.1 - left.0;
                let m_right = right.1 - right.0;

                let left_eval_2 = left.1 + m_left;
                let left_eval_3 = left_eval_2 + m_left;

                let right_eval_2 = right.1 + m_right;
                let right_eval_3 = right_eval_2 + m_right;

                println!(
                    "dense E2 partial evasls: {:?}",
                    [
                        [eq_evals.0, eq_evals.1, eq_evals.2],
                        [left.0, left_eval_2, left_eval_3],
                        [right.0, right_eval_2, right_eval_3]
                    ]
                );
                (
                    eq_evals.0 * left.0 * right.0,
                    eq_evals.1 * left_eval_2 * right_eval_2,
                    eq_evals.2 * left_eval_3 * right_eval_3,
                )
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
            )
    } else {
        // ---------------- Dao–Thaler mode: Eq(i_A,i_C,i_B) = E2(i_A,i_B) * E1(i_C) ----------------
        //
        // Here we treat:
        //   - E1: inner dimension over C (columns),
        //   - E2: outer dimension over A|B (rows).
        //
        // For each row (fixed i_A,i_B), we:
        //   1. combine the relevant E1 entries with P-chunks along the C direction,
        //   2. multiply the resulting inner sum by the corresponding E2(row) value.
        let E1_len = eq_poly.E1_len;

        // Precompute Dao–Thaler E1 evaluations at the three needed points j ∈ {0,2,3}
        // for each C-position (i_C). E1_evals[c] = (E1(c, j=0), E1(c, j=2), E1(c, j=3)).
        let E1_evals: Vec<_> = eq_poly.E1[..E1_len]
            .par_chunks(2)
            .map(|E1_chunk| {
                let eval_point_0 = E1_chunk[0];
                let m_eq = E1_chunk[1] - E1_chunk[0];
                let eval_point_2 = E1_chunk[1] + m_eq;
                let eval_point_3 = eval_point_2 + m_eq;
                (eval_point_0, eval_point_2, eval_point_3)
            })
            .collect();

        // The poly currently represents `poly.len() / 2` Eq points (each point
        // corresponds to 2 coefficients in an interleaved L/R representation).
        //
        // This worker is logically responsible for `worker_len` Eq points starting
        // at `global_start`, but we must not read beyond what `poly` actually has.
        let slice_end = eq_poly.global_start + core::cmp::min(eq_poly.len, poly.len() / 2);

        // Upper bound (exclusive) on E2 indices this worker can actually use:
        //
        //   - A row with global index r covers global Eq indices [r * E1_len, (r+1)*E1_len),
        //   - we only care about rows that intersect [global_start, slice_end),
        //   - convert that intersection into a local row-offset range for this worker.
        let E2_local_bound = slice_end
                .div_ceil(E1_len) // first row index strictly after slice_end
                .saturating_sub(eq_poly.row_start)
                .min(eq_poly.E2_len);

        // Dao–Thaler outer loop: iterate over each relevant row of E2 and perform
        // the inner sum over the C dimension, restricted to this worker’s slice.
        eq_poly.E2[..E2_local_bound]
            .par_iter()
            .enumerate()
            .map(|(E2_i, E2_eval)| {
                // Global row index in the full Eq table.
                let r = eq_poly.row_start + E2_i;

                // Global Eq index range covered by this row: [row_first, row_last).
                let row_first = r * E1_len;
                let row_last = row_first + E1_len;

                // Intersection with the worker’s assigned slice [global_start, worker_end),
                // expressed in global Eq indices.
                let eq_first = eq_poly.global_start.max(row_first);
                let eq_last = (eq_poly.global_start + eq_poly.len).min(row_last);

                // We expect this row to intersect the slice if it is within E2_local_bound.
                debug_assert!(eq_last > eq_first);

                // Column offsets inside the row (in Eq points).
                let col_from = eq_first - row_first;
                let col_to = eq_last - row_first;

                // Each Dao–Thaler E1 entry spans 2 Eq points; enforce alignment.
                debug_assert!(
                    col_from % 2 == 0 && col_to % 2 == 0,
                    "misaligned Eq slice within row"
                );

                // Range of C-indices (pairs) in this row that belong to this worker.
                let E1_from = col_from / 2;
                let E1_to = col_to / 2;

                // Local Eq point index inside this worker’s slice:
                //
                //   local_point_idx = eq_first - global_start
                //
                // Each point corresponds to 2 coefficients in the interleaved polynomial.
                let poly_from = (eq_first - eq_poly.global_start) * 2;
                assert!(poly_from < poly.len(), "coeff_start out of bounds");

                let mut inner_sum = (F::zero(), F::zero(), F::zero());

                // Inner Dao–Thaler sum along C:
                //
                //   sum_{c in [pair_from,pair_to)} E1_evals[c] * P_chunk(c)
                //
                // where P_chunk(c) is the 4-coefficient interleaved block for that
                // position in the grand product GKR wiring.
                for (E1_evals, chunk) in E1_evals[E1_from..E1_to]
                    .iter()
                    .zip(poly.coeffs[poly_from..poly.len()].chunks(4))
                {
                    let left = (
                        *chunk.first().unwrap_or(&F::zero()),
                        *chunk.get(2).unwrap_or(&F::zero()),
                    );
                    let right = (
                        *chunk.get(1).unwrap_or(&F::zero()),
                        *chunk.get(3).unwrap_or(&F::zero()),
                    );

                    let m_left = left.1 - left.0;
                    let m_right = right.1 - right.0;

                    let left_eval_2 = left.1 + m_left;
                    let left_eval_3 = left_eval_2 + m_left;

                    let right_eval_2 = right.1 + m_right;
                    let right_eval_3 = right_eval_2 + m_right;

                    println!(
                        "dense E1 partial evasls: {:?}",
                        [
                            [
                                E1_evals.0 * *E2_eval,
                                E1_evals.1 * *E2_eval,
                                E1_evals.2 * *E2_eval
                            ],
                            [left.0, left_eval_2, left_eval_3],
                            [right.0, right_eval_2, right_eval_3]
                        ]
                    );

                    inner_sum.0 += E1_evals.0 * left.0 * right.0;
                    inner_sum.1 += E1_evals.1 * left_eval_2 * right_eval_2;
                    inner_sum.2 += E1_evals.2 * left_eval_3 * right_eval_3;
                }

                // Multiply by the outer E2(row) value for this (i_A, i_B) row.
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

#[cfg(test)]
fn sparse_interleaved_sumcheck_evals<F: JoltField>(
    poly: &SparseInterleavedPolynomial<F>,
    eq_poly: &DistributedSplitEqPolynomial<F>,
    worker: usize,
) -> Vec<F> {
    use crate::field::OptimizedMul;

    if let Some(coalesced) = &poly.coalesced {
        return dense_interleaved_sumcheck_evals(&coalesced, eq_poly);
    }
    println!("sparse_interleaved_sumcheck_evals");

    // We use the Dao-Thaler optimization for the EQ polynomial, so there are two cases we
    // must handle. For details, refer to Section 2.2 of https://eprint.iacr.org/2024/1210.pdf
    let cubic_evals = if eq_poly.E1_len == 1 {
        // If `eq_poly.E1` has been fully bound, we compute the cubic polynomial as we
        // would without the Dao-Thaler optimization, using the standard linear-time
        // sumcheck algorithm with optimizations for sparsity.

        let eq_evals: Vec<[F; 3]> = eq_poly
            .E2
            .par_chunks(2)
            .take(poly.dense_len / 4)
            .map(|eq_chunk| {
                let eval_point_0 = eq_chunk[0];
                let m_eq = eq_chunk[1] - eq_chunk[0];
                let eval_point_2 = eq_chunk[1] + m_eq;
                let eval_point_3 = eval_point_2 + m_eq;
                [eval_point_0, eval_point_2, eval_point_3]
            })
            .collect();
        // This is what \sum_{x} eq(r, x) * left(x) * right(x) would be if
        // `left` and `right` were both all ones.
        let eq_eval_sums: [F; 3] = eq_evals
            .par_iter()
            .fold(
                || [F::zero(); 3],
                |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
            )
            .reduce(
                || [F::zero(), F::zero(), F::zero()],
                |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
            );
        // Now we compute the deltas, correcting `eq_eval_sums` for the
        // elements of `left` and `right` that aren't ones.
        let deltas: [F; 3] = poly
            .coeffs
            .par_iter()
            .flat_map(|segment| {
                segment
                    .par_chunk_by(|x, y| x.index / 4 == y.index / 4)
                    .map(|sparse_block| {
                        let block_index = sparse_block[0].index / 4;
                        let mut block = [F::one(); 4];
                        for coeff in sparse_block {
                            block[coeff.index % 4] = coeff.value;
                        }

                        let left = (block[0], block[2]);
                        let right = (block[1], block[3]);

                        let m_left = left.1 - left.0;
                        let m_right = right.1 - right.0;

                        let left_eval_2 = left.1 + m_left;
                        let left_eval_3 = left_eval_2 + m_left;

                        let right_eval_2 = right.1 + m_right;
                        let right_eval_3 = right_eval_2 + m_right;

                        let eq_evals = eq_evals[block_index];

                        println!(
                            "E2 partial evals: {:?}",
                            [
                                eq_evals,
                                [left.0, left_eval_2, left_eval_3],
                                [right.0, right_eval_2, right_eval_3]
                            ]
                        );
                        println!("-------");

                        [
                            eq_evals[0].mul_0_optimized(left.0.mul_1_optimized(right.0) - F::one()),
                            eq_evals[1] * (left_eval_2 * right_eval_2 - F::one()),
                            eq_evals[2] * (left_eval_3 * right_eval_3 - F::one()),
                        ]
                    })
            })
            .reduce(
                || [F::zero(); 3],
                |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
            );

        println!("--------------");
        [
            eq_eval_sums[0] + deltas[0],
            eq_eval_sums[1] + deltas[1],
            eq_eval_sums[2] + deltas[2],
        ]
    } else {
        // This is a more complicated version of the `else` case in
        // `DenseInterleavedPolynomial::compute_cubic`. Read that one first.

        // We start by computing the E1 evals:
        // (1 - j) * E1[0, x1] + j * E1[1, x1]
        let E1_evals: Vec<_> = eq_poly.E1[..eq_poly.E1_len]
            .par_chunks(2)
            .map(|E1_chunk| {
                let eval_point_0 = E1_chunk[0];
                let m_eq = E1_chunk[1] - E1_chunk[0];
                let eval_point_2 = E1_chunk[1] + m_eq;
                let eval_point_3 = eval_point_2 + m_eq;
                [eval_point_0, eval_point_2, eval_point_3]
            })
            .collect();
        // Now compute \sum_{x1} ((1 - j) * E1[0, x1] + j * E1[1, x1])
        let E1_eval_sums = E1_evals
            .par_iter()
            .fold(
                || [F::zero(); 3],
                |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
            )
            .reduce(
                || [F::zero(); 3],
                |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
            );

        let num_x1_bits = eq_poly.E1_len.log_2() - 1;
        let x1_bitmask = (1 << num_x1_bits) - 1;

        // Iterate over the non-one coefficients and compute the deltas (relative to
        // what the cubic would be if all the coefficients were ones).
        let deltas = poly
            .coeffs
            .par_iter()
            .flat_map(|segment| {
                segment
                    .par_chunk_by(|a, b| {
                        // Group by x2
                        let a_x2 = (a.index / 4) >> num_x1_bits;
                        let b_x2 = (b.index / 4) >> num_x1_bits;
                        a_x2 == b_x2
                    })
                    .map(|chunk| {
                        let mut inner_sum = [F::zero(); 3];
                        let x2 = (chunk[0].index / 4) >> num_x1_bits;

                        for sparse_block in chunk.chunk_by(|x, y| x.index / 4 == y.index / 4) {
                            let block_index = sparse_block[0].index / 4;
                            let mut block = [F::one(); 4];
                            for coeff in sparse_block {
                                block[coeff.index % 4] = coeff.value;
                            }

                            let left = (block[0], block[2]);
                            let right = (block[1], block[3]);

                            let m_left = left.1 - left.0;
                            let m_right = right.1 - right.0;

                            let left_eval_2 = left.1 + m_left;
                            let left_eval_3 = left_eval_2 + m_left;

                            let right_eval_2 = right.1 + m_right;
                            let right_eval_3 = right_eval_2 + m_right;

                            let x1 = block_index & x1_bitmask;
                            let delta = [
                                E1_evals[x1][0]
                                    .mul_0_optimized(left.0.mul_1_optimized(right.0) - F::one()),
                                E1_evals[x1][1] * (left_eval_2 * right_eval_2 - F::one()),
                                E1_evals[x1][2] * (left_eval_3 * right_eval_3 - F::one()),
                            ];

                            println!(
                                "E1 deltas: {:?}",
                                [
                                    E1_evals[x1].map(|e| e * eq_poly.E2[x2]),
                                    [left.0, left_eval_2, left_eval_3],
                                    [right.0, right_eval_2, right_eval_3]
                                ]
                            );
                            println!("-------");
                            inner_sum[0] += delta[0];
                            inner_sum[1] += delta[1];
                            inner_sum[2] += delta[2];
                        }

                        println!("--------------");

                        [
                            eq_poly.E2[x2] * inner_sum[0],
                            eq_poly.E2[x2] * inner_sum[1],
                            eq_poly.E2[x2] * inner_sum[2],
                        ]
                    })
            })
            .reduce(
                || [F::zero(); 3],
                |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
            );

        println!("deltas aggr: {:?}", deltas);
        println!("-------");

        // The cubic evals assuming all the coefficients are ones is affected by the
        // `dense_len`, since we implicitly 0-pad the `dense_len` to a power of 2.
        //
        // As a refresher, the cubic evals we're computing are:
        //
        // \sum_{x2} E2[x2] * (\sum_{x1} ((1 - j) * E1[0, x1] + j * E1[1, x1]) * \prod_k ((1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2)))
        let evals_assuming_all_ones = if poly.dense_len.is_power_of_two() {
            // If `dense_len` is a power of 2, there is no 0-padding.
            //
            // So we have:
            // \sum_{x2} (E2[x2] * (\sum_{x1} ((1 - j) * E1[0, x1] + j * E1[1, x1]) * 1))
            //   = \sum_{x2} (E2[x2] * \sum_{x1} E1_evals[x1])
            //   = (\sum_{x2} E2[x2]) * (\sum_{x1} E1_evals[x1])
            //   = 1 * E1_eval_sums
            println!("E1_eval_sums: {:?}", E1_eval_sums);
            println!("-------");

            if worker == 0 {
                E1_eval_sums
            } else {
                [F::zero(); 3]
            }
        } else {
            let chunk_size = poly.dense_len.next_power_of_two() / eq_poly.E2_len;
            let num_all_one_chunks = poly.dense_len / chunk_size;
            let E2_sum: F = eq_poly.E2[..num_all_one_chunks].iter().sum();
            if poly.dense_len % chunk_size == 0 {
                // If `dense_len` isn't a power of 2 but evenly divides `chunk_size`,
                // that means that for the last values of x2, we have:
                //   (1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2)) = 0
                // due to the 0-padding.
                //
                // This makes the entire inner sum 0 for those values of x2.
                // So we can simply sum over E2 for the _other_ values of x2, and
                // multiply by `E1_eval_sums`.

                println!(
                    "Eq all ones % chunk_size: {:?}",
                    E1_eval_sums.map(|e| e * E2_sum)
                );
                println!("-------");

                E1_eval_sums.map(|e| e * E2_sum)
            } else {
                // If `dense_len` isn't a power of 2 and doesn't divide `chunk_size`,
                // the last nonzero "chunk" will have (self.dense_len % chunk_size) ones,
                // followed by (chunk_size - self.dense_len % chunk_size) zeros,
                // e.g. 1 1 1 1 1 1 1 1 0 0 0 0
                //
                // This handles this last chunk:
                let last_chunk_evals = E1_evals[..(poly.dense_len % chunk_size) / 4]
                    .par_iter()
                    .fold(
                        || [F::zero(); 3],
                        |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
                    )
                    .reduce(
                        || [F::zero(); 3],
                        |sum, evals| [sum[0] + evals[0], sum[1] + evals[1], sum[2] + evals[2]],
                    );

                println!(
                    "Eq all ones !% chunk_size: {:?}",
                    [
                        E2_sum * E1_eval_sums[0]
                            + eq_poly.E2[num_all_one_chunks] * last_chunk_evals[0],
                        E2_sum * E1_eval_sums[1]
                            + eq_poly.E2[num_all_one_chunks] * last_chunk_evals[1],
                        E2_sum * E1_eval_sums[2]
                            + eq_poly.E2[num_all_one_chunks] * last_chunk_evals[2],
                    ]
                );
                println!("-------");
                [
                    E2_sum * E1_eval_sums[0] + eq_poly.E2[num_all_one_chunks] * last_chunk_evals[0],
                    E2_sum * E1_eval_sums[1] + eq_poly.E2[num_all_one_chunks] * last_chunk_evals[1],
                    E2_sum * E1_eval_sums[2] + eq_poly.E2[num_all_one_chunks] * last_chunk_evals[2],
                ]
            }
        };

        println!("--------------");

        [
            evals_assuming_all_ones[0] + deltas[0],
            evals_assuming_all_ones[1] + deltas[1],
            evals_assuming_all_ones[2] + deltas[2],
        ]
    };

    cubic_evals.to_vec()
}

#[test]
fn test_memories_allocation() {
    type F = ark_bn254::Fr;
    const NUM_MEMORIES: usize = 51;

    let m: usize = env::var("CHUNK_SIZE")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap();

    let W: usize = env::var("NUM_WORKERS")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .unwrap();

    let M: usize = env::var("M")
        .unwrap_or_else(|_| "65536".to_string())
        .parse()
        .unwrap();

    assert!(m.is_power_of_two(), "chunk_size must be a power of two");
    assert!(W.is_power_of_two(), "num_workers must be a power of two");

    println!("NUM_WORKERS={} | CHUNK_SIZE={}", W, m);

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
            let read_ct = vec![F::from(read_index); m];
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

            (subtable, final_cts)
        })
        .unzip();
    // izip!(&materialized_subtables, &final_cts)
    //     .for_each(|(subtable, final_cts)| println!("init: {:?} final: {:?}", subtable, final_cts));
    let final_cts = final_cts.into_iter().flatten().collect_vec();

    println!("\n/---------- Construct layers ----------/");

    let mut read_write_workers = vec![vec![]; W];
    let mut flags_workers = vec![vec![]; W];
    let mut init_final_workers = vec![vec![]; W];

    let mut rng = test_rng();

    for w in 0..W {
        let worker_rw_memories = read_write_memories_for_worker(NUM_MEMORIES, W, w);
        let w_chunk_len = worker_rw_memories.len() * 2 * m;

        println!(
            "worker {} read_write_memories [{}]: {:?} w_chunk_len: {}",
            w,
            worker_rw_memories.len(),
            worker_rw_memories,
            w_chunk_len
        );

        read_write_workers[w] = worker_rw_memories
            .iter()
            .flat_map(|&memory_index| {
                let read_cts = &read_cts[memory_index];

                let read_fingerprints: Vec<_> = (0..m).map(|i| read_cts[i]).collect();
                let write_fingerprints: Vec<_> = read_fingerprints
                    .iter()
                    .map(|read_fingerprint| *read_fingerprint + F::ONE)
                    .collect();

                [read_fingerprints, write_fingerprints]
            })
            .collect();

        flags_workers[w] = worker_rw_memories
            .iter()
            .flat_map(|_| {
                let flags = (0..m)
                    .filter_map(|i| {
                        if rng.gen_range(0..3) == 0 {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                [flags.clone(), flags]
            })
            .collect();

        println!("---------");

        let worker_memories_for_subtables =
            init_final_subtables_for_worker(&subtable_to_memory_indices, W, w);
        let final_memories = worker_memories_for_subtables
            .iter()
            .flat_map(|(_, memories)| memories)
            .copied()
            .collect_vec();
        println!(
            "worker {} memories_for_subtables [{}]: {:?}",
            w,
            final_memories.len(),
            worker_memories_for_subtables
        );

        init_final_workers[w] = worker_memories_for_subtables
            .into_iter()
            .flat_map(|(subtable_index, memories)| {
                let subtable = subtable_index.map(|si| &materialized_subtables[si]);
                let mut leaves_len = M * memories.len();
                if subtable.is_some() {
                    leaves_len += M;
                }
                let mut leaves = vec![F::ZERO; leaves_len];
                let mut leaf_index = 0;

                // Init leaves
                if let Some(subtable) = subtable {
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
        read_write_workers.iter().map(|wp| wp.len()).sum::<usize>(),
        NUM_MEMORIES * 2 * m
    );

    println!("/--------------- READ_WRITE ---------------/");

    run_simulation_sparse_dbgp_batch_wize(
        m,
        NUM_MEMORIES * 2,
        W,
        read_write_workers,
        flags_workers,
    );

    println!("/------------------------------------------/");

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

    println!("/--------------- INIT_FINAL ---------------/");

    run_simulation_dbgp_batch_wize(M, 75, W, init_final_workers);

    println!("/------------------------------------------/");
}

#[cfg(test)]
fn run_simulation_sparse_dbgp_batch_wize<F: JoltField>(
    chunk_size: usize,
    N: usize,
    W: usize,
    w_fingerprints: Vec<Vec<Vec<F>>>,
    w_flags: Vec<Vec<Vec<usize>>>,
) {
    use crate::subprotocols::sparse_grand_product::BatchedGrandProductToggleLayer;

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

    let mut in_interleaved = vec![];

    for i in 1..N + 1 {
        in_interleaved.extend(vec![F::from(i as u64); chunk_size]);
    }
    // println!("in_interleaved: {:?}", in_interleaved);

    for w in 0..W {
        println!("worker {} flags: {:?}", w, w_flags[w]);
    }

    let W_log2 = W.log_2();
    let leaves_len = chunk_size * N;
    let num_layers = (leaves_len / N).log_2();
    println!("num_layers: {}", num_layers);

    #[derive(Debug, Clone)]
    struct LayerCircuit<F: JoltField> {
        layer_idx: usize,
        polys: Vec<SparseInterleavedPolynomial<F>>,
    }

    struct LayerProof<F: JoltField> {
        proof: SumcheckInstanceProof<F, KeccakTranscript>,
        left_claim: F,
        right_claim: F,
    }

    println!("\n/---------- Construct layers ----------/");

    let input_layer = LayerCircuit {
        layer_idx: num_layers,
        polys: izip!(w_fingerprints.iter().cloned(), w_flags.iter().cloned())
            .map(|(fingerprints, flags)| {
                let toggle = BatchedGrandProductToggleLayer::new(flags, fingerprints);
                toggle.layer_output()
            })
            .collect(),
    };

    // for w in 0..W {
    //     println!(
    //         "worker {} input layer ({}) | polys [{}]: {:?}",
    //         w,
    //         input_layer.layer_idx,
    //         input_layer.polys[w].len(),
    //         input_layer.polys[w].coeffs
    //     );
    // }

    let mut worker_layers = vec![input_layer];

    for i in 0..num_layers - 1 {
        let prev_layer = &worker_layers[i];
        let layer_idx = num_layers - i - 1;

        let polys = prev_layer
            .polys
            .par_iter()
            .map(SparseInterleavedPolynomial::layer_output)
            .collect::<Vec<_>>();

        let next_layer = LayerCircuit { layer_idx, polys };

        worker_layers.push(next_layer);
    }

    let grand_product_output = {
        let last_layer = worker_layers.last().unwrap();
        last_layer
            .polys
            .par_iter()
            .flat_map(|poly| {
                let (left, right) = poly.uninterleave();
                izip!(&left, &right).map(|(a, b)| *a * *b).collect_vec()
            })
            .collect::<Vec<_>>()
    };

    println!("grand product output {:?}", grand_product_output);

    let mut layer_proofs = vec![];
    let mut transcript = KeccakTranscript::new(&[]);

    //------ Output layer (N) prover
    let output_mle = DensePolynomial::new_padded(grand_product_output.clone());
    let mut num_rounds = output_mle.get_num_vars();
    let mut r_grand_product: Vec<_> = transcript.challenge_vector(num_rounds);
    // gkr output claim, will be updated as output claim of each subsequent layer as we progress to input layer
    let mut grand_product_claim = output_mle.evaluate(&r_grand_product);

    //------ Distributed layers prover

    println!("\n/------------ Worker prover ------------/");

    let batch_sizes_check = (0..W)
        .map(|w| {
            let batch_size = worker_layers[0].polys[w].dense_len / chunk_size;
            if w == W - 1 {
                (N - batch_size) / (W - 1)
            } else {
                batch_size
            }
        })
        .collect::<Vec<_>>();

    assert!(batch_sizes_check.iter().all(|&b| b == batch_sizes_check[0]));
    let mut eq_pairs_per_worker = batch_sizes_check[0];

    for mut layer in worker_layers.iter().cloned().rev() {
        // println!(
        //     "layer {} rounds: {:?} polys: {:?}",
        //     layer.layer_idx,
        //     num_rounds,
        //     layer.polys.iter().map(|p| &p.coeffs).collect::<Vec<_>>()
        // );

        let mut eq_polys = (0..W)
            .map(|w| {
                DistributedSplitEqPolynomial::new(&r_grand_product, W_log2, w, eq_pairs_per_worker)
            })
            .collect_vec();

        let (proof, r_sumcheck, (left_claim, right_claim)) =
            SumcheckInstanceProof::<F, KeccakTranscript>::simulate_prove_cubic_distributed_sparse_batch_wize(
                &grand_product_claim,
                num_rounds,
                &mut eq_polys,
                &r_grand_product,
                &mut layer.polys,
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
        eq_pairs_per_worker *= 2;
        println!("---------------------------");
    }

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
}

#[cfg(test)]
fn run_simulation_dbgp_batch_wize<F: JoltField>(
    chunk_size: usize,
    N: usize,
    W: usize,
    w_interleaved: Vec<Vec<F>>,
) {
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

    let mut in_interleaved = vec![];

    for i in 1..N + 1 {
        in_interleaved.extend(vec![F::from(i as u64); chunk_size]);
    }
    // println!("in_interleaved: {:?}", in_interleaved);

    let W_log2 = W.log_2();
    let leaves_len = chunk_size * N;
    let num_layers = (leaves_len / N).log_2();
    println!("num_layers: {}", num_layers);

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

    let input_layer = LayerCircuit {
        layer_idx: num_layers,
        polys: w_interleaved
            .iter()
            .cloned()
            .map(DenseInterleavedPolynomial::new)
            .collect(),
    };

    // for w in 0..W {
    //     println!(
    //         "worker {} input layer ({}) | polys [{}]: {:?}",
    //         w,
    //         input_layer.layer_idx,
    //         input_layer.polys[w].len(),
    //         input_layer.polys[w].coeffs
    //     );
    // }

    let mut worker_layers = vec![input_layer];

    for i in 0..num_layers - 1 {
        let prev_layer = &worker_layers[i];
        let layer_idx = num_layers - i - 1;

        let polys = prev_layer
            .polys
            .par_iter()
            .map(DenseInterleavedPolynomial::layer_output)
            .collect::<Vec<_>>();

        let next_layer = LayerCircuit { layer_idx, polys };

        worker_layers.push(next_layer);
    }

    let grand_product_output = {
        let last_layer = worker_layers.last().unwrap();
        last_layer
            .polys
            .par_iter()
            .flat_map(|poly| {
                let (left, right) = poly.uninterleave();
                izip!(&left, &right).map(|(a, b)| *a * *b).collect_vec()
            })
            .collect::<Vec<_>>()
    };

    println!("grand product output {:?}", grand_product_output);

    let mut layer_proofs = vec![];
    let mut transcript = KeccakTranscript::new(&[]);

    //------ Output layer (N) prover
    let output_mle = DensePolynomial::new_padded(grand_product_output.clone());
    let mut num_rounds = output_mle.get_num_vars();
    let mut r_grand_product: Vec<_> = transcript.challenge_vector(num_rounds);
    // gkr output claim, will be updated as output claim of each subsequent layer as we progress to input layer
    let mut grand_product_claim = output_mle.evaluate(&r_grand_product);

    //------ Distributed layers prover

    println!("\n/------------ Worker prover ------------/");

    let batch_sizes_check = (0..W)
        .map(|w| {
            let batch_size = worker_layers[0].polys[w].len() / chunk_size;
            if w == W - 1 {
                (N - batch_size) / (W - 1)
            } else {
                batch_size
            }
        })
        .collect::<Vec<_>>();

    assert!(batch_sizes_check.iter().all(|&b| b == batch_sizes_check[0]));
    let mut eq_pairs_per_worker = batch_sizes_check[0];

    for mut layer in worker_layers.iter().cloned().rev() {
        // println!(
        //     "layer {} rounds: {:?} polys: {:?}",
        //     layer.layer_idx,
        //     num_rounds,
        //     layer.polys.iter().map(|p| &p.coeffs).collect::<Vec<_>>()
        // );

        let mut eq_polys = (0..W)
            .map(|w| {
                DistributedSplitEqPolynomial::new(&r_grand_product, W_log2, w, eq_pairs_per_worker)
            })
            .collect_vec();

        let (proof, r_sumcheck, (left_claim, right_claim)) =
            SumcheckInstanceProof::<F, KeccakTranscript>::simulate_prove_cubic_distributed_batch_wize(
                &grand_product_claim,
                num_rounds,
                &mut eq_polys,
                &r_grand_product,
                &mut layer.polys,
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
        eq_pairs_per_worker *= 2;
        println!("---------------------------");
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
fn ran_local_sprase_gkr_simulation<F: JoltField>(
    chunk_size: usize,
    N: usize,
    fingerprints: Vec<Vec<F>>,
    flags: Vec<Vec<usize>>,
) {
    use crate::subprotocols::sparse_grand_product::BatchedGrandProductToggleLayer;

    println!("CHUNK_SIZE={}; BATCH_SIZE={}", chunk_size, N);

    let leaves_len = chunk_size * N;
    let num_layers = (leaves_len / N).log_2();
    println!("num_layers: {}", num_layers);

    struct LayerCircuit<F: JoltField> {
        layer_idx: usize,
        poly: SparseInterleavedPolynomial<F>,
    }

    struct LayerProof<F: JoltField> {
        proof: SumcheckInstanceProof<F, KeccakTranscript>,
        left_claim: F,
        right_claim: F,
    }

    println!("\n/----------- Construct layers -----------/");

    let input_layer = LayerCircuit {
        layer_idx: num_layers,
        poly: {
            let toggle = BatchedGrandProductToggleLayer::new(flags, fingerprints);
            toggle.layer_output()
        },
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
        println!("poly: {:?}", layer.poly.dense_len);

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

/// Compute per-worker delta in *chunks* for given batch_size (in chunks) and num_workers (power of 2).
///
/// Old element-based version was:
///   N_worker = floor(N / W)
///   t = floor(log2(N_worker))
///   M_elems = 2^(t + L - 1)
///   P_layer = N_worker * 2^L
///   delta_elems = P_layer mod M_elems, with sign rule:
///       if delta_elems == M_elems/2 -> +delta_elems
///       else                         -> -delta_elems
///
/// Dividing by 2^L (chunk_size), this simplifies in *chunks* to:
///   M_chunks = 2^(t-1)
///   delta_chunks_base = N_worker mod M_chunks
///   if delta_chunks_base == 0      -> 0
///   else if delta_chunks_base == M_chunks/2 -> +delta_chunks_base
///   else                           -> -delta_chunks_base
pub fn calculate_delta_per_worker(batch_size: usize, num_workers: usize) -> isize {
    assert!(num_workers > 0 && num_workers.is_power_of_two());

    // N_worker = floor(N / W)
    let n_worker = batch_size / num_workers;
    if n_worker == 0 {
        return 0;
    }

    // t = floor(log2(N_worker))
    let t = (usize::BITS - 1 - n_worker.leading_zeros()) as u32;

    // For t == 0 or 1, the original element-wise delta is always 0.
    if t <= 1 {
        return 0;
    }

    let m_chunks: usize = 1usize << (t - 1); // 2^(t-1)
    let delta_base: usize = n_worker % m_chunks; // in chunks

    if delta_base == 0 {
        return 0;
    }

    let half: usize = m_chunks >> 1; // 2^(t-2)

    // if delta_base == half {
    //     delta_base as isize // +delta
    // } else {
    //      // -delta
    // }
    -(delta_base as isize)
}

/// Given:
/// - num_memories = M (each memory = 2 chunks),
/// - num_workers = W (power of 2),
/// split the big polynomial [0 .. 2*M chunks) among workers using the delta trick,
/// and return which memories this `worker_idx` touches.
///
/// The big poly is in *chunks*:
///   N = 2 * M
///   N_worker = floor(N / W)
///   delta_chunks = calculate_delta_per_worker(N, W)
/// Non-last workers get `base_len_chunks = N_worker + delta_chunks` chunks;
/// last worker gets the remainder.
/// A memory i occupies chunks [2*i, 2*i + 2).
pub fn read_write_memories_for_worker(
    num_memories: usize,
    num_workers: usize,
    worker_idx: usize,
) -> Vec<usize> {
    assert!(num_memories > 0);
    assert!(num_workers > 0 && num_workers.is_power_of_two());
    assert!(worker_idx < num_workers);

    // Total chunks and per-worker baseline
    let n_chunks = 2 * num_memories; // N = M * 2
    let n_worker = n_chunks / num_workers; // floor(N/W)
    assert!(n_worker > 0, "not enough chunks per worker");

    // Shared delta in *chunks*
    let delta_chunks = calculate_delta_per_worker(n_chunks, num_workers);

    // Length (in chunks) of a non-last worker's portion
    let base_len_chunks = (n_worker as isize + delta_chunks) as usize;
    assert!(
        base_len_chunks > 0,
        "non-last worker chunk_len must be positive"
    );

    let total_chunks = n_chunks;

    // Compute this worker's chunk range [start_chunk, end_chunk)
    let (start_chunk, end_chunk) = if worker_idx + 1 < num_workers {
        let start = base_len_chunks * worker_idx;
        let end = start + base_len_chunks;
        (start, end)
    } else {
        // last worker gets the remainder
        let start = base_len_chunks * (num_workers - 1);
        let end = total_chunks;
        (start, end)
    };

    // Each memory i occupies chunks [2*i, 2*i + 2)
    let mut memories = Vec::new();
    for mem_idx in 0..num_memories {
        let mem_start = 2 * mem_idx;
        let mem_end = mem_start + 2;
        // non-empty intersection with [start_chunk, end_chunk)
        if mem_start < end_chunk && mem_end > start_chunk {
            memories.push(mem_idx);
        }
    }

    memories
}

/// For a given worker, return a sequence of "segments" in the global memory layout:
/// - `Option<usize>` is `Some(subtable_idx)` if the worker owns the header block of that subtable,
///   or `None` if it only owns some memories from that subtable.
/// - `Vec<usize>` are the memory indices (from subtable_to_memory_indices) that fall into this
///   worker's polynomial slice.
///
/// Layout in *blocks/chunks*:
///   for each subtable i:
///       [header_block] + [mem_block_0] + [mem_block_1] + ...
///
/// Splitting:
///   B = total_blocks
///   N = B
///   N_worker = floor(N / num_workers)
///   delta_chunks = calculate_delta_per_worker(N, num_workers)
///   non-last workers:  len_chunks = N_worker + delta_chunks
///   last worker:        len_chunks = N - len_chunks * (num_workers - 1)
///
/// A header of subtable i is at block index `pref[i]`.
/// A memory j in subtable i is at block index `pref[i] + 1 + j`.
pub fn init_final_subtables_for_worker(
    subtable_to_memory_indices: &[Vec<usize>],
    num_workers: usize, // power of two
    worker_idx: usize,
) -> Vec<(Option<usize>, Vec<usize>)> {
    assert!(
        num_workers > 0 && num_workers.is_power_of_two(),
        "num_workers must be power of two"
    );
    assert!(worker_idx < num_workers, "worker_idx out of bounds");

    // Prefix sums in *block* space: each subtable contributes 1 header + |st| memory blocks.
    let mut pref = Vec::with_capacity(subtable_to_memory_indices.len() + 1);
    pref.push(0usize);
    for st in subtable_to_memory_indices {
        pref.push(pref.last().copied().unwrap() + 1 + st.len());
    }
    let total_blocks = *pref.last().unwrap(); // B == N
    assert!(total_blocks > 0, "no blocks to allocate");

    let batch_size = total_blocks; // N
    let n_worker = batch_size / num_workers; // floor(N / W)
    assert!(n_worker > 0, "not enough blocks per worker");

    // delta in *chunks/blocks*
    let delta_chunks = calculate_delta_per_worker(batch_size, num_workers);

    // Length of a non-last worker's slice, in blocks.
    let base_len_chunks = (n_worker as isize + delta_chunks) as usize;
    assert!(
        base_len_chunks > 0,
        "non-last worker slice must be positive"
    );

    let total_chunks = batch_size; // one chunk per block

    // Chunk range [start_chunk, end_chunk) for this worker.
    let (start_chunk, end_chunk) = if worker_idx + 1 < num_workers {
        let start = base_len_chunks * worker_idx;
        let end = start + base_len_chunks;
        (start, end)
    } else {
        let start = base_len_chunks * (num_workers - 1);
        let end = total_chunks;
        (start, end)
    };

    // Map chunk interval [start_chunk, end_chunk) back to per-subtable segments.
    let mut out: Vec<(Option<usize>, Vec<usize>)> = Vec::new();

    for (i, st) in subtable_to_memory_indices.iter().enumerate() {
        let st_beg_block = pref[i];
        let st_end_block = pref[i + 1];

        if st_beg_block >= end_chunk {
            break; // past this worker's range
        }
        if st_end_block <= start_chunk {
            continue; // entirely before this worker's range
        }

        // Header block index
        let header_block = st_beg_block;
        let header_in_range = header_block >= start_chunk && header_block < end_chunk;

        // Memory blocks
        let mems_beg_block = st_beg_block + 1;
        let mut mems_for_worker = Vec::new();

        for (j, &mem_id) in st.iter().enumerate() {
            let blk = mems_beg_block + j;
            if blk >= end_chunk {
                break; // remaining mems from this subtable are beyond this worker
            }
            if blk >= start_chunk {
                mems_for_worker.push(mem_id);
            }
        }

        // Include subtable if either header or at least one memory is in range.
        if header_in_range || !mems_for_worker.is_empty() {
            let header_tag = if header_in_range { Some(i) } else { None };
            out.push((header_tag, mems_for_worker));
        }
    }

    out
}

#[cfg(test)]
fn debug_workers_data<F: JoltField>(chunk_size: usize, N: usize, W: usize) -> Vec<Vec<F>> {
    assert!(
        chunk_size.is_power_of_two(),
        "chunk_size must be a power of two"
    );
    assert!(W.is_power_of_two(), "num_workers must be a power of two");
    let N_worker = N / W; // floor(N / W)

    let mut in_interleaved = vec![];

    for i in 1..N + 1 {
        in_interleaved.extend(vec![F::from(i as u64); chunk_size]);
    }
    println!("in_interleaved: {:?}", in_interleaved);

    let mut w_interleaved = vec![vec![]; W];
    let mut leaves = in_interleaved.clone();
    let delta = if W == 1 {
        0
    } else {
        calculate_delta_per_worker(N, W) * chunk_size as isize
    };
    println!("batch_size_worker: {} delta: {}", N_worker, delta);
    for w in 0..W {
        let mut w_chunk_len = ((chunk_size * N_worker) as isize + delta) as usize;
        if w == W - 1 {
            w_chunk_len = N * chunk_size - w_chunk_len * (W - 1);
        };
        println!("worker: {} w_chunk_len: {}", w, w_chunk_len);

        w_interleaved[w] = leaves.drain(0..w_chunk_len).collect();
    }
    assert!(leaves.is_empty());
    w_interleaved
}

#[test]
fn test_distributed_sparse_gkr_simulation() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();
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

    let mut rng = test_rng();
    let mut flags = (0..N / 2)
        .flat_map(|_| {
            let flags = (0..chunk_size)
                .filter_map(|i| {
                    if rng.gen_range(0..3) == 0 {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            [flags.clone(), flags]
        })
        .collect_vec();
    let (w_fingerprints, w_flags) = debug_workers_data(chunk_size, N, W)
        .into_iter()
        .map(|w| {
            let fingerprints = w.chunks(chunk_size).map(|c| c.to_vec()).collect_vec();
            let flags = flags.drain(..fingerprints.len()).collect_vec();
            let flags = flags.chunks(2).map(|c| c[1].clone()).collect_vec();
            (fingerprints, flags)
        })
        .unzip();

    run_simulation_sparse_dbgp_batch_wize::<ark_bn254::Fr>(
        chunk_size,
        N,
        W,
        w_fingerprints,
        w_flags,
    )
}

#[test]
fn test_local_sparse_gkr_simulation() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();
    let chunk_size: usize = env::var("CHUNK_SIZE")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap();
    let N: usize = env::var("BATCH_SIZE")
        .unwrap_or_else(|_| "4".to_string())
        .parse()
        .unwrap();

    let mut rng = test_rng();
    let flags = (0..N / 2)
        .map(|_| {
            let mut flags = (0..chunk_size)
                .filter_map(|i| {
                    if rng.gen_range(0..3) == 0 {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            if flags.is_empty() {
                flags.push(1);
            }
            flags.clone()
        })
        .collect_vec();
    println!("flags: {:?}", flags);
    let fingerprints = debug_workers_data(chunk_size, N, 1)[0]
        .chunks(chunk_size)
        .map(|c| c.to_vec())
        .collect_vec();

    ran_local_sprase_gkr_simulation::<ark_bn254::Fr>(chunk_size, N, fingerprints, flags);
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

    let w_interleaved = debug_workers_data(chunk_size, N, W);
    run_simulation_dbgp_batch_wize::<ark_bn254::Fr>(chunk_size, N, W, w_interleaved);
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
        let w_interleaved = debug_workers_data(chunk_size, N, W);

        run_simulation_dbgp_batch_wize::<ark_bn254::Fr>(chunk_size, N, W, w_interleaved);
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
