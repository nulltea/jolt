#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::jolt::subtable::eq;
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
use std::env;
use std::marker::PhantomData;
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

    /// prove_arbitrary ((LOW to HIGH))
    #[cfg(test)]
    pub fn prove_arbitrary_low_to_high<Func>(
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
            // println!("round {}--------------", _round);
            // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
            // for points {0, ..., |g(x)|}
            println!(
                "round {} | left/right: {}",
                _round,
                polys[1].len(), // &polys[1].coeffs_as_field_elements()[..polys[1].len()]
            );
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
                                BindingOrder::LowToHigh,
                            )
                        })
                        .collect();
                    for j in 0..combined_degree {
                        let evals_j: Vec<_> = evals.iter().map(|x| x[j]).collect();
                        accum[j] += comb_func(&evals_j);
                    }
                    // println!(
                    //     "round: {} poly_term_i: {} evals: {:?}",
                    //     _round, poly_term_i, evals
                    // );

                    accum
                })
                .collect();

            // println!("accum: {:?}", accum);

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

            // println!("eval points: {:?}", eval_points);
            // println!("------");

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
                .for_each(|poly| poly.bind(r_j, BindingOrder::LowToHigh));
            previous_claim = univariate_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        let final_evals = polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
    }

    #[cfg(test)]
    pub fn simulate_distibuted_prove_arbitrary<Func>(
        claim: &F,
        num_rounds: usize,
        workers_polys: &mut [Vec<MultilinearPolynomial<F>>],
        chunk_size_per_worker: usize,
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

            let mle_half = workers_polys[0][1].len() / 2;

            let mut eval_points = workers_polys
                .par_iter()
                .enumerate()
                .map(|(worker, polys)| {
                    let mut eval_points = vec![F::zero(); combined_degree];

                    if worker == 0 {
                        println!(
                            "round {} | left/right {:?}",
                            _round,
                            polys[1].len(), // &polys[1].coeffs_as_field_elements()[..polys[1].len()]
                        );
                    }
                    let accum: Vec<Vec<F>> = (0..mle_half)
                        .into_par_iter()
                        .map(|poly_term_i| {
                            let mut accum = vec![F::zero(); combined_degree];

                            // TODO(moodlezoup): Optimize
                            let evals: Vec<_> = polys
                                .iter()
                                .enumerate()
                                .map(|(poly_i, poly)| {
                                    if poly_i == 0 {
                                        custom_eq_sumcheck_evals(
                                            poly,
                                            poly_term_i,
                                            combined_degree,
                                            global_eq_pairs,
                                            worker,
                                            num_workers,
                                        )
                                    } else {
                                        poly.sumcheck_evals(
                                            poly_term_i,
                                            combined_degree,
                                            BindingOrder::LowToHigh,
                                        )
                                    }
                                })
                                .collect();
                            // println!(
                            //     "round: {} poly_term_i: {} evals: {:?}",
                            //     _round,
                            //     poly_term_i + mle_half * worker,
                            //     evals
                            // );
                            for j in 0..combined_degree {
                                let evals_j: Vec<_> = evals.iter().map(|x| x[j]).collect();
                                accum[j] += comb_func(&evals_j);
                            }

                            accum
                        })
                        .collect();

                    // println!("accum: {:?}", accum);

                    eval_points
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(poly_i, eval_point)| {
                            *eval_point += accum
                                .par_iter()
                                .take(mle_half)
                                .map(|mle| mle[poly_i])
                                .sum::<F>();
                        });
                    // println!("------");
                    eval_points
                })
                .reduce(
                    || vec![F::zero(); combined_degree],
                    |mut eval_points, eval_points_next| {
                        izip!(eval_points.iter_mut(), eval_points_next).for_each(|(a, b)| *a += b);
                        eval_points
                    },
                );

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

            for polys in workers_polys.iter_mut() {
                // bound all tables to the verifier's challenge
                (*polys)
                    .par_iter_mut()
                    .for_each(|poly| poly.bind(r_j, BindingOrder::LowToHigh));
                previous_claim = univariate_poly.evaluate(&r_j);
            }
            compressed_polys.push(compressed_poly);
        }

        let remaining_evals: Vec<Vec<Vec<F>>> = workers_polys
            .iter()
            .map(|polys| {
                polys
                    .iter()
                    .map(|poly| poly.coeffs_as_field_elements()[..poly.len()].to_vec())
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut polys: Vec<MultilinearPolynomial<F>> = (0..workers_polys[0].len())
            .map(|i| {
                if i == 0 {
                    MultilinearPolynomial::from(remaining_evals[0][i].to_vec())
                } else {
                    MultilinearPolynomial::from(interleave_n(
                        (0..num_workers)
                            .map(|w| remaining_evals[w][i].to_vec())
                            .collect_vec(),
                    ))
                }
            })
            .collect();

        // println!(
        //     "polys: {:?}",
        //     polys.iter().map(|p| p.len()).collect::<Vec<_>>()
        // );

        let remaining_rounds = num_rounds - worker_rounds;

        println!("remaining rounds: {}", remaining_rounds);

        for _round in 0..remaining_rounds {
            println!(
                "rem round {} | left/right: {}",
                _round,
                polys[1].len(), // &polys[1].coeffs_as_field_elements()[..polys[1].len()]
            );

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
                                BindingOrder::LowToHigh,
                            )
                        })
                        .collect();

                    // println!(
                    //     "round: {} poly_term_i: {} evals: {:?}",
                    //     _round, poly_term_i, evals
                    // );
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

            // println!("remaining evals: {:?}", eval_points);

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
                .for_each(|poly| poly.bind(r_j, BindingOrder::LowToHigh));
            previous_claim = univariate_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        // println!(
        //     "polys: {:?}",
        //     polys.iter().map(|poly| poly.len()).collect::<Vec<_>>()
        // );

        let final_evals = polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
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
/// - `BLOCK = CHUNK_PER_WORKER / W` (must divide evenly)
/// - For a worker `w` and local index `i`, let `k = i / BLOCK` and `off = i % BLOCK`.
/// - The corresponding global index is `g = (k*W + w)*BLOCK + off`.
///
/// We then perform the standard LowToHigh sumcheck evaluation at `g`:
/// `evals[j] = P(2g + j)` for `j` on the univariate degree points.
fn custom_eq_sumcheck_evals<F: JoltField>(
    poly: &MultilinearPolynomial<F>,
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
    evals[0] = poly.get_bound_coeff(2 * global_index);
    if degree == 1 {
        return evals;
    }
    let mut eval = poly.get_bound_coeff(2 * global_index + 1);
    let m = eval - evals[0];
    for i in 1..degree {
        eval += m;
        evals[i] = eval;
    }
    evals
}

#[cfg(test)]
fn run_distributed_gkr_simulation<F: JoltField>(chunk_size: usize, N: usize, W: usize) {
    assert!(
        chunk_size.is_power_of_two(),
        "chunk_size must be a power of two"
    );
    assert!(W.is_power_of_two(), "num_workers must be a power of two");
    println!(
        "NUM_WORKERS={} | CHUNK_SIZE={}={}/worker | BATCH_SIZE={}",
        W,
        chunk_size,
        chunk_size / W,
        N
    );

    let W_log2 = W.log_2();
    let leaves_len = chunk_size * N;
    let num_layers = (leaves_len / N).log_2();
    println!("num_layers: {}", num_layers);

    let LR_SIZE = chunk_size / 2;
    let LR_SIZE_worker = LR_SIZE / W;

    assert!(LR_SIZE_worker > 1, "chunk_size must be greater than 2");

    // Batch is as follows: [[1,1,1,...,1],[2,2,2,...,2],[3,3,3,...,3],...] each chunk has length `chunk_size`
    // So left/right polys are L(x): [[1,...,1],[2,...,2],[3,...,3],...]; R(x): [[1,...,1],[2,...,2],[3,...,3],...] each chunk has length `chunk_size/2`
    // The expected grand product results are thus [1^chunk_size, 2^chunk_size, 3^chunk_size, ...]
    let mut in_left = vec![1; LR_SIZE]
        .into_iter()
        .map(F::from)
        .collect::<Vec<_>>();
    let mut in_right = vec![1; LR_SIZE]
        .into_iter()
        .map(F::from)
        .collect::<Vec<_>>();

    for i in 2..N + 1 {
        in_left.extend(vec![F::from(i as u64); LR_SIZE]);
        in_right.extend(vec![F::from(i as u64); LR_SIZE]);
    }
    println!("left: {:?} right: {:?}", in_left, in_right);

    #[derive(Debug, Clone)]
    struct LayerCircuit<F: JoltField> {
        layer_idx: usize,
        left: Vec<Vec<F>>,
        right: Vec<Vec<F>>,
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
    let mut input_layer = LayerCircuit {
        layer_idx: num_layers,
        left: (0..W).map(|_| vec![F::from(1); LR_SIZE_worker]).collect(),
        right: (0..W).map(|_| vec![F::from(1); LR_SIZE_worker]).collect(),
    };

    for i in 2..N + 1 {
        for w in 0..W {
            input_layer.left[w].extend(vec![F::from(i as u64); LR_SIZE_worker]);
            input_layer.right[w].extend(vec![F::from(i as u64); LR_SIZE_worker]);
        }
    }

    println!(
        "input layer ({}) | left: {:?} right: {:?}",
        input_layer.layer_idx, input_layer.left, input_layer.right
    );

    let worker_num_layers = num_layers - 2 - W_log2;
    let mut worker_layers = vec![input_layer];

    for i in 0..worker_num_layers {
        let prev_layer = &worker_layers[i];
        let layer_idx = num_layers - i - 1;

        let outputs = (0..W)
            .map(|w| {
                izip!(&prev_layer.left[w], &prev_layer.right[w])
                    .map(|(a, b)| *a * *b)
                    .collect::<Vec<F>>()
            })
            .collect_vec();
        println!("worker layer {} | out: {:?}", num_layers - i, outputs);
        println!("-------------");
        let (left, right): (Vec<_>, Vec<_>) = (0..W).map(|w| uninterleave(&outputs[w])).unzip();

        // `MultilinearPolynomial::sumcheck_evals`'s the semantic pairing is (2i, 2i+1) in the coefficient array.
        // Left/right polys must contain at least 2 elements from each chunk,
        // otherwise `sumcheck_evals` would be pairing elements from unrelated chunks; breaking the distribution invariance.
        // TODO: if you comment this assert, and set W_log2:=1 verification passes for W>2; need to investigate this.
        assert!(
            left[0].len() == right[0].len() && left[0].len() > N,
            "Left/right polys must contain at least 2 elements from each chunk"
        );

        let next_layer = LayerCircuit {
            layer_idx,
            left,
            right,
        };

        println!(
            "worker layer {} | left: {:?} right {:?}",
            layer_idx, next_layer.left, next_layer.right
        );
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

        let prev_outputs = if !switched {
            let prev_layer = worker_layers.last().unwrap();

            // What coordinator receives from workers (sub hashes)
            let prev_outputs_by_worker = (0..W)
                .map(|w| {
                    izip!(&prev_layer.left[w], &prev_layer.right[w])
                        .map(|(a, b)| *a * *b)
                        .collect::<Vec<F>>()
                })
                .collect_vec();

            println!(
                "worker layer {} | out: {:?}",
                num_layers - worker_num_layers,
                prev_outputs_by_worker
            );
            println!("-------------");

            // We restore order of results like this:
            // Worker results in our toy example: [[1, 4, 9, 16], [1, 4, 9, 16], ...]
            // The restored order is: [1,..1, 4...4, 9...9, 16...16] (interleave_n)
            let prev_outputs = interleave_n(
                (0..W)
                    .into_iter()
                    .map(|w| prev_outputs_by_worker[w].chunks(1).collect_vec())
                    .collect::<Vec<_>>(),
            )
            .into_par_iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
            println!(
                "worker layer {} | out restore order: {:?}",
                num_layers - worker_num_layers,
                prev_outputs
            );
            switched = true;
            prev_outputs
        } else {
            // After switch run normally
            let prev_layer = coordinator_layers.last().unwrap();

            let prev_outputs = izip!(&prev_layer.left[0], &prev_layer.right[0])
                .map(|(a, b)| *a * *b)
                .collect::<Vec<_>>();

            println!(
                "coordinator layer {} | out: {:?}",
                prev_layer.layer_idx, prev_outputs
            );
            prev_outputs
        };
        println!("-------------");

        let (left, right) = uninterleave(&prev_outputs);

        let next_layer = LayerCircuit {
            layer_idx,
            left: vec![left],
            right: vec![right],
        };

        println!(
            "coordinator layer {} | left: {:?} right {:?}",
            layer_idx, next_layer.left, next_layer.right
        );
        coordinator_layers.push(next_layer);
    }

    let grand_product_output = {
        let last_layer = coordinator_layers.last().unwrap();

        izip!(&last_layer.left[0], &last_layer.right[0])
            .map(|(a, b)| *a * *b)
            .collect::<Vec<_>>()
    };

    println!("gkr output {:?}", grand_product_output);

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
    for layer in coordinator_layers.into_iter().rev() {
        println!("layer 1 rounds: {} -------", num_rounds);

        let left_poly = MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
            layer.left[0].clone(),
        ));
        let right_poly = MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
            layer.right[0].clone(),
        ));

        let eq_poly = MultilinearPolynomial::<F>::from(EqPolynomial::evals(&r_grand_product));

        let mut polys = vec![eq_poly, left_poly, right_poly];
        let (proof, r_sumcheck, final_evals) =
            SumcheckInstanceProof::<F, KeccakTranscript>::prove_arbitrary_low_to_high(
                &grand_product_claim,
                num_rounds,
                &mut polys,
                |evals| evals[0] * evals[1] * evals[2],
                3,
                &mut transcript,
            );

        let left_claim = final_evals[1];
        let right_claim = final_evals[2];

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
    for layer in worker_layers.iter().cloned().rev() {
        println!("layer {} rounds: {:?} -------", layer.layer_idx, num_rounds);
        println!("in_left: {:?}", layer.left);
        println!("in_right: {:?}", layer.right);

        let eq_poly = MultilinearPolynomial::<F>::from(EqPolynomial::evals(&r_grand_product));

        let mut workers_polys = (0..W)
            .map(|w| {
                vec![
                    eq_poly.clone(),
                    MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
                        layer.left[w].clone(),
                    )),
                    MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
                        layer.right[w].clone(),
                    )),
                ]
            })
            .collect_vec();

        let (proof, r_sumcheck, final_evals) =
            SumcheckInstanceProof::<F, KeccakTranscript>::simulate_distibuted_prove_arbitrary(
                &grand_product_claim,
                num_rounds,
                &mut workers_polys,
                chunk_size_per_worker,
                |evals| evals[0] * evals[1] * evals[2],
                3,
                &mut transcript,
            );
        let left_claim = final_evals[1];
        let right_claim = final_evals[2];

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

    let chunk_size_worker = chunk_size / W;
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

    let partial_openings: Vec<_> = izip!(
        worker_layers[0].left.clone(),
        worker_layers[0].right.clone()
    )
    .map(|(w_left, w_right)| {
        let w_opennings = izip!(
            w_left.chunks(LR_SIZE_worker),
            w_right.chunks(LR_SIZE_worker)
        )
        .map(|(l_chunk, r_chunk)| {
            let w_chunk = interleave(l_chunk.to_vec(), r_chunk.to_vec()).collect_vec();
            assert_eq!(w_chunk.len(), chunk_size_worker);
            MultilinearPolynomial::from(w_chunk).evaluate(&r_opening_worker)
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
    izip!(
        in_left.chunks(LR_SIZE),
        in_right.chunks(LR_SIZE),
        prover_openings
    )
    .for_each(|(l_coeffs, r_coeffs, prover_opening)| {
        let coeffs = interleave(l_coeffs.to_vec(), r_coeffs.to_vec()).collect_vec();
        assert_eq!(coeffs.len(), chunk_size);
        let verifier_opening = MultilinearPolynomial::from(coeffs).evaluate(&r_opening);
        assert_eq!(verifier_opening, prover_opening);
    });
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

    println!(
        "CHUNK_SIZE={}/per_worker={}; BATCH_SIZE={}",
        chunk_size,
        chunk_size / 2,
        N
    );

    let leaves_len = chunk_size * N;
    let num_layers = (leaves_len / N).log_2();
    println!("num_layers: {}", num_layers);

    let LR_SIZE = chunk_size / 2;

    let mut in_left = vec![1; LR_SIZE]
        .into_iter()
        .map(F::from)
        .collect::<Vec<_>>();
    let mut in_right = vec![1; LR_SIZE]
        .into_iter()
        .map(F::from)
        .collect::<Vec<_>>();

    for i in 2..N + 1 {
        in_left.extend(vec![F::from(i as u64); LR_SIZE]);
        in_right.extend(vec![F::from(i as u64); LR_SIZE]);
    }
    println!("left: {:?} right: {:?}", in_left, in_right);

    struct LayerCircuit<F: JoltField> {
        layer_idx: usize,
        left: Vec<Vec<F>>,
        right: Vec<Vec<F>>,
    }

    struct LayerProof<F: JoltField> {
        proof: SumcheckInstanceProof<F, KeccakTranscript>,
        left_claim: F,
        right_claim: F,
    }

    println!("\n/----------- Construct layers -----------/");

    let input_layer = LayerCircuit {
        layer_idx: num_layers,
        left: vec![in_left.clone()],
        right: vec![in_right.clone()],
    };

    println!(
        "input layer | left: {:?} right: {:?}",
        input_layer.left, input_layer.right
    );

    let mut layers = vec![input_layer];

    for i in 0..num_layers - 1 {
        let prev_layer = &layers[i];
        let prev_outputs = izip!(&prev_layer.left[0], &prev_layer.right[0])
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>();

        println!(
            "layer {} | prev_outputs: {:?}",
            num_layers - i - 1,
            prev_outputs
        );

        let layer_idx = num_layers - i - 1;

        let (left, right) = uninterleave(&prev_outputs);

        let next_layer = LayerCircuit {
            layer_idx,
            left: vec![left],
            right: vec![right],
        };

        println!(
            "layer {} | left: {:?} right {:?}",
            layer_idx, next_layer.left, next_layer.right
        );
        layers.push(next_layer);
    }

    let last_layer = layers.last().unwrap();
    let grand_product_output = izip!(last_layer.left[0].clone(), last_layer.right[0].clone())
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

    for layer in layers.into_iter().rev() {
        println!("layer {} rounds: {:?}-------", layer.layer_idx, num_rounds);
        println!("in_left: {:?}", layer.left);
        println!("in_right: {:?}", layer.right);
        println!("r_grand_product: {:?}", r_grand_product);

        let left_poly = MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
            layer.left[0].clone(),
        ));
        let right_poly = MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
            layer.right[0].clone(),
        ));

        let eq_poly = MultilinearPolynomial::<F>::from(EqPolynomial::evals(&r_grand_product));

        let mut polys = vec![eq_poly, left_poly, right_poly];
        let (proof, r_sumcheck, final_evals) =
            SumcheckInstanceProof::<F, KeccakTranscript>::prove_arbitrary_low_to_high(
                &grand_product_claim,
                num_rounds,
                &mut polys,
                |evals| evals[0] * evals[1] * evals[2],
                3,
                &mut transcript,
            );

        let left_claim = final_evals[1];
        let right_claim = final_evals[2];

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

    let prover_openings = izip!(in_left.chunks(LR_SIZE), in_right.chunks(LR_SIZE))
        .map(|(l_coeffs, r_coeffs)| {
            let coeffs = interleave(l_coeffs.to_vec(), r_coeffs.to_vec()).collect_vec();
            assert_eq!(coeffs.len(), chunk_size);
            MultilinearPolynomial::from(coeffs).evaluate(&r_opening)
        })
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
    izip!(
        in_left.chunks(LR_SIZE),
        in_right.chunks(LR_SIZE),
        prover_openings
    )
    .for_each(|(l_coeffs, r_coeffs, prover_opening)| {
        let coeffs = interleave(l_coeffs.to_vec(), r_coeffs.to_vec()).collect_vec();
        assert_eq!(coeffs.len(), chunk_size);
        let verifier_opening = MultilinearPolynomial::from(coeffs).evaluate(&r_opening);
        assert_eq!(verifier_opening, prover_opening);
    });
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

    run_distributed_gkr_simulation::<ark_bn254::Fr>(chunk_size, N, W);
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
        (1 << 13, 102), // read_write
        (1 << 16, 75),  // init_final
    ];
    for (chunk_size, N) in cases {
        println!("/------------ Test case start ------------/");
        run_distributed_gkr_simulation::<ark_bn254::Fr>(chunk_size, N, W);
        println!("/-----------------------------------------/");
    }
}

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
