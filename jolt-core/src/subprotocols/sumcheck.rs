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
            // println!("round {}--------------", _round);
            // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
            // for points {0, ..., |g(x)|}
            println!(
                "round {} | left: {} right: {}",
                _round,
                polys[1].len(), // &polys[1].coeffs_as_field_elements()[..polys[1].len()]
                polys[2].len()  // &polys[2].coeffs_as_field_elements()[..polys[2].len()]
            );
            let mut eval_points = vec![F::zero(); combined_degree];

            let mle_half = polys[0].len() / 2;

            let accum: Vec<Vec<F>> = (0..mle_half)
                .into_iter()
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
                    println!(
                        "round: {} poly_term_i: {} evals: {:?}",
                        _round, poly_term_i, evals
                    );

                    accum
                })
                .collect();

            println!("accum: {:?}", accum);

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

            println!("eval points: {:?}", eval_points);
            println!("------");

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

    #[tracing::instrument(skip_all, name = "Sumcheck.prove", level = "trace")]
    pub fn simulate_distibuted_prove_arbitrary<Func>(
        claim: &F,
        num_rounds: usize,
        polys: &mut [&mut Vec<MultilinearPolynomial<F>>],
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

        // #[cfg(test)]
        // {
        //     let total_evals = 1 << num_rounds;
        //     let mut sum = F::zero();
        //     for i in 0..total_evals {
        //         let params: Vec<F> = polys.iter().map(|poly| poly.get_coeff(i)).collect();
        //         sum += comb_func(&params);
        //     }
        //     assert_eq!(&sum, claim, "Sumcheck claim is wrong");
        // }

        fn custom_eq_sumcheck_evals<F: JoltField>(
            poly: &MultilinearPolynomial<F>,
            mut index: usize,
            degree: usize,
            half_mle: usize,
            chunk_mle: usize,
            worker: usize,
        ) -> Vec<F> {
            assert_ne!(chunk_mle, 1);

            let mle_by2 = (0..half_mle)
                .chunks(chunk_mle / 2)
                .into_iter()
                .map(|chunk| chunk.collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let (even, odd) = uninterleave(&mle_by2);
            let mle_rewired = [even, odd][worker]
                .iter()
                .cloned()
                .flatten()
                .collect::<Vec<_>>();
            // println!("worker {} mle: {:?}", worker, mle_);
            index = mle_rewired[index];

            let mut evals = vec![F::zero(); degree];
            evals[0] = poly.get_bound_coeff(2 * index);
            if degree == 1 {
                return evals;
            }
            let mut eval = poly.get_bound_coeff(2 * index + 1);
            let m = eval - evals[0];
            for i in 1..degree {
                eval += m;
                evals[i] = eval;
            }
            evals
        }

        let chunk_nv = chunk_size_per_worker.log_2();
        let mut chunk_mle = chunk_size_per_worker;

        for _round in 0..chunk_nv {
            // println!("round {}--------------", _round);

            // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
            // for points {0, ..., |g(x)|}
            let mut eval_points = vec![F::zero(); combined_degree];

            let mle_half = polys[0][1].len() / 2;
            let eq_poly_half_mle = mle_half * 2;

            for (worker, polys) in polys.iter().enumerate() {
                if worker == 0 {
                    println!(
                        "round {} | left {:?} right {:?}",
                        _round,
                        polys[1].len(), // &polys[1].coeffs_as_field_elements()[..polys[1].len()]
                        polys[2].len()  // &polys[2].coeffs_as_field_elements()[..polys[2].len()]
                    );
                }
                let accum: Vec<Vec<F>> = (0..mle_half)
                    .into_iter()
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
                                        eq_poly_half_mle,
                                        chunk_mle,
                                        worker,
                                    )
                                } else {
                                    poly.sumcheck_evals(
                                        poly_term_i,
                                        combined_degree,
                                        BindingOrder::LowToHigh,
                                    )
                                }
                                // poly.sumcheck_evals(
                                //     poly_term_i,
                                //     combined_degree,
                                //     BindingOrder::LowToHigh,
                                // )
                            })
                            .collect();
                        println!(
                            "round: {} poly_term_i: {} evals: {:?}",
                            _round,
                            poly_term_i + mle_half * worker,
                            evals
                        );
                        for j in 0..combined_degree {
                            let evals_j: Vec<_> = evals.iter().map(|x| x[j]).collect();
                            accum[j] += comb_func(&evals_j);
                        }

                        accum
                    })
                    .collect();

                println!("accum: {:?}", accum);

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
            }

            chunk_mle /= 2;

            println!("eval points: {:?}", eval_points);

            // println!("--------------");

            eval_points.insert(1, previous_claim - eval_points[0]);
            let univariate_poly = UniPoly::from_evals(&eval_points);
            let compressed_poly = univariate_poly.compress();
            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            for polys in polys.iter_mut() {
                // bound all tables to the verifier's challenge
                (*polys)
                    .par_iter_mut()
                    .for_each(|poly| poly.bind(r_j, BindingOrder::LowToHigh));
                previous_claim = univariate_poly.evaluate(&r_j);
            }
            compressed_polys.push(compressed_poly);
        }

        let final_evals: Vec<Vec<Vec<F>>> = polys
            .iter()
            .map(|polys| {
                polys
                    .iter()
                    .map(|poly| poly.coeffs_as_field_elements()[..poly.len()].to_vec())
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut polys: Vec<MultilinearPolynomial<F>> = (0..polys[0].len())
            .map(|i| {
                if i == 0 {
                    MultilinearPolynomial::from(final_evals[0][i].to_vec())
                } else {
                    MultilinearPolynomial::from(
                        interleave(final_evals[0][i].to_vec(), final_evals[1][i].to_vec())
                            .collect::<Vec<_>>(),
                    )
                }
            })
            .collect();

        // println!(
        //     "polys: {:?}",
        //     polys.iter().map(|p| p.len()).collect::<Vec<_>>()
        // );

        let remaining_rounds = num_rounds - chunk_nv;

        println!("remaining rounds: {}", remaining_rounds);

        for _round in 0..remaining_rounds {
            // println!("remaining round {}--------------", _round);
            println!(
                "rem round {} | left: {} right: {}",
                _round,
                polys[1].len(), // &polys[1].coeffs_as_field_elements()[..polys[1].len()]
                polys[2].len(), // &polys[2].coeffs_as_field_elements()[..polys[2].len()]
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

                    println!(
                        "round: {} poly_term_i: {} evals: {:?}",
                        _round, poly_term_i, evals
                    );
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

            println!("remaining evals: {:?}", eval_points);

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

        println!(
            "polys: {:?}",
            polys.iter().map(|poly| poly.len()).collect::<Vec<_>>()
        );

        let final_evals = polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
    }
}

#[test]
fn test_distributed_gkr_simulation() {
    type F = ark_bn254::Fr;

    let CHUNK_SIZE: usize = env::var("CHUNK_SIZE")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap();
    let N: usize = env::var("BATCH_SIZE")
        .unwrap_or_else(|_| "4".to_string())
        .parse()
        .unwrap();

    println!(
        "CHUNK_SIZE={}/per_worker={}; BATCH_SIZE={}",
        CHUNK_SIZE,
        CHUNK_SIZE / 2,
        N
    );

    let leaves_len = CHUNK_SIZE * N;
    let num_layers = (leaves_len / N).log_2();
    println!("num_layers: {}", num_layers);

    let K = CHUNK_SIZE / 2;
    let K_worker = K / 2;

    let mut in_left = vec![1; K].into_iter().map(F::from).collect::<Vec<_>>();
    let mut in_right = vec![1; K].into_iter().map(F::from).collect::<Vec<_>>();

    for i in 1..N {
        in_left.extend(
            vec![1; K as usize]
                .into_iter()
                .map(|e| F::from(e) + F::from(i as u64))
                .collect::<Vec<_>>(),
        );
        in_right.extend(
            vec![1; K as usize]
                .into_iter()
                .map(|e| F::from(e) + F::from(i as u64))
                .collect::<Vec<_>>(),
        );
    }
    println!("left: {:?} right: {:?}", in_left, in_right);

    // Normal scenario
    {
        println!("Local scenario ----------------");

        let layer2 = interleave(in_left.clone(), in_right.clone()).collect_vec();
        println!("input layer: {:?}", layer2);

        let mut layer = layer2;
        for i in 0..num_layers - 1 {
            layer = layer.chunks(2).map(|c| c[0] * c[1]).collect_vec();
            println!("layer {}: {:?}", num_layers - i - 1, layer);
            let (left, right) = uninterleave(&layer);
            println!(
                "layer {}: left {:?} right {:?}",
                num_layers - i - 1,
                left,
                right
            );
        }

        let output = layer.chunks(2).map(|c| c[0] * c[1]).collect_vec();
        println!("local output: {:?}", output);
        println!("-------------------------------");
    }

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

    println!("\nConstruct layers----------------");

    let mut input_layer = LayerCircuit {
        layer_idx: num_layers,
        left: vec![
            vec![1; K_worker]
                .into_iter()
                .map(F::from)
                .collect::<Vec<_>>(),
            vec![1; K_worker]
                .into_iter()
                .map(F::from)
                .collect::<Vec<_>>(),
        ],
        right: vec![
            vec![1; K_worker]
                .into_iter()
                .map(F::from)
                .collect::<Vec<_>>(),
            vec![1; K_worker]
                .into_iter()
                .map(F::from)
                .collect::<Vec<_>>(),
        ],
    };

    // println!(
    //     "input layer_ | left: {:?} right: {:?}",
    //     input_layer.left, input_layer.right
    // );

    for i in 1..N {
        input_layer.left[0].extend(
            vec![1; K_worker as usize]
                .into_iter()
                .map(|e| F::from(e) + F::from(i as u64))
                .collect::<Vec<_>>(),
        );
        input_layer.left[1].extend(
            vec![1; K_worker as usize]
                .into_iter()
                .map(|e| F::from(e) + F::from(i as u64))
                .collect::<Vec<_>>(),
        );

        input_layer.right[0].extend(
            vec![1; K_worker as usize]
                .into_iter()
                .map(|e| F::from(e) + F::from(i as u64))
                .collect::<Vec<_>>(),
        );
        input_layer.right[1].extend(
            vec![1; K_worker as usize]
                .into_iter()
                .map(|e| F::from(e) + F::from(i as u64))
                .collect::<Vec<_>>(),
        );
    }

    println!(
        "input layer ({}) | left: {:?} right: {:?}",
        input_layer.layer_idx, input_layer.left, input_layer.right
    );

    let mut layers = vec![input_layer];

    // let num_layers = 2;

    let distributed_layers = num_layers - 3;

    for i in 0..distributed_layers {
        let prev_layer = &layers[i];
        let layer_idx = num_layers - i - 1;

        let outputs = [
            izip!(&prev_layer.left[0], &prev_layer.right[0])
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>(),
            izip!(&prev_layer.left[1], &prev_layer.right[1])
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>(),
        ];
        println!("worker layer {} | out: {:?}", num_layers - i, outputs);
        println!("-------------");
        let (left0, right0) = uninterleave(&outputs[0]);
        let (left1, right1) = uninterleave(&outputs[1]);

        let next_layer = LayerCircuit {
            layer_idx,
            left: vec![left0, left1],
            right: vec![right0, right1],
        };

        println!(
            "worker layer {} | left: {:?} right {:?}",
            layer_idx, next_layer.left, next_layer.right
        );
        layers.push(next_layer);
    }

    let mut coordinator_layers = vec![];

    for i in 0..1 {
        let layer_idx = num_layers - distributed_layers - i - 1;

        let prev_layer = layers.last().unwrap();

        // What coordinator receives from workers (sub hashes)
        let prev_outputs_by_worker = [
            izip!(&prev_layer.left[0], &prev_layer.right[0])
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>(),
            izip!(&prev_layer.left[1], &prev_layer.right[1])
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>(),
        ];

        println!(
            "worker layer {} | out: {:?}",
            num_layers - distributed_layers,
            prev_outputs_by_worker
        );
        println!("-------------");

        let prev_outputs_by_worker: Vec<_> = vec![
            prev_outputs_by_worker[0].clone(),
            prev_outputs_by_worker[1].clone(),
        ];

        let prev_outputs = interleave(
            prev_outputs_by_worker[0].chunks(2),
            prev_outputs_by_worker[1].chunks(2),
        )
        .flatten()
        .copied()
        .collect_vec();

        println!(
            "worker layer {} | out perm: {:?}",
            num_layers - distributed_layers,
            prev_outputs
        );

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

    let gkr_output = {
        // let prev_layer = layers.last().unwrap();
        let prev_layer = coordinator_layers.last().unwrap();

        // What coordinator receives from workers (sub hashes)
        // let (left, right) = (
        //     izip!(&prev_layer.left[0], &prev_layer.right[0])
        //         .map(|(a, b)| a * b)
        //         .collect::<Vec<_>>(),
        //     izip!(&prev_layer.left[1], &prev_layer.right[1])
        //         .map(|(a, b)| a * b)
        //         .collect::<Vec<_>>(),
        // );

        // println!(
        //     "coordinator layer {} | out: {:?}",
        //     num_layers - 2,
        //     [left.clone(), right.clone()]
        // );

        let prev_outputs = izip!(&prev_layer.left[0], &prev_layer.right[0])
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>();

        println!(
            "coordinator layer {} | out: {:?}",
            prev_layer.layer_idx, prev_outputs
        );
        println!("-------------");

        // println!("pre_last_outputs (sigma): {:?}", pre_last_outputs);

        let (left, right) = uninterleave(&prev_outputs);
        println!("output layer | left {:?} right {:?}", left, right);
        let gkr_output = izip!(&left, &right).map(|(a, b)| a * b).collect::<Vec<_>>();

        coordinator_layers.push(LayerCircuit {
            layer_idx: 1,
            left: vec![left],
            right: vec![right],
        });
        gkr_output
    };

    println!("gkr output {:?}", gkr_output);

    println!("\nCoordinator prover----------------");

    //------ Output layer (N) prover
    let mut layer_proofs = vec![];
    let mut transcript = KeccakTranscript::new(&[]);

    let output_mle = DensePolynomial::new_padded(gkr_output.clone());
    let mut num_rounds = output_mle.get_num_vars();
    let mut r_grand_product: Vec<_> = transcript.challenge_vector(num_rounds);
    let mut running_claim = output_mle.evaluate(&r_grand_product);

    for layer in coordinator_layers.into_iter().rev() {
        println!("layer 1 rounds: {}", num_rounds);

        let left_poly = MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
            layer.left[0].clone(),
        ));
        let right_poly = MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
            layer.right[0].clone(),
        ));

        let eq_poly = MultilinearPolynomial::<F>::from(EqPolynomial::evals(&r_grand_product));

        let mut polys = vec![eq_poly, left_poly, right_poly];
        let (proof, r_sumcheck, final_evals) =
            SumcheckInstanceProof::<F, KeccakTranscript>::prove_arbitrary(
                &running_claim,
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
        running_claim = left_claim + r_layer * (right_claim - left_claim);

        println!("layer {} claim: {:?}", layer.layer_idx, running_claim);

        r_grand_product = r_sumcheck.iter().rev().copied().collect();
        r_grand_product.push(r_layer); // pass r_grand_product to next layer
        num_rounds += 1;
    }

    //------ Distributed layers prover

    println!("\nWorker prover--------------");

    let mut chunk_size_per_worker = 2;
    for layer in layers.into_iter().rev() {
        println!("layer {} rounds: {:?}-------", layer.layer_idx, num_rounds);
        println!("in_left: {:?}", layer.left);
        println!("in_right: {:?}", layer.right);

        println!("r_grand_product: {:?}", r_grand_product);

        // {
        //     let outputs = [
        //         izip!(&layer.left[0], &layer.right[0])
        //             .map(|(a, b)| a * b)
        //             .collect::<Vec<_>>(),
        //         izip!(&layer.left[1], &layer.right[1])
        //             .map(|(a, b)| a * b)
        //             .collect::<Vec<_>>(),
        //     ];

        //     let prev_outputs: Vec<_> = if layer.layer_idx == 2 {
        //         interleave(outputs[0].clone(), outputs[1].clone()).collect()
        //     } else {
        //         [outputs[0].clone(), outputs[1].clone()].concat()
        //     };

        //     println!("prev_outputs: {:?}", prev_outputs);

        //     let claim_check = DensePolynomial::new_padded(prev_outputs).evaluate(&r_grand_product);
        //     assert_eq!(claim_check, running_claim);
        // }

        let eq_poly = MultilinearPolynomial::<F>::from(EqPolynomial::evals(&r_grand_product));

        let in_left0_poly = MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
            layer.left[0].clone(),
        ));
        let in_right0_poly = MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
            layer.right[0].clone(),
        ));
        // let eq_poly2_0 = MultilinearPolynomial::<F>::from(eq_evals[0..eq_evals.len() / 2].to_vec());
        let mut polys1 = vec![eq_poly.clone(), in_left0_poly, in_right0_poly];
        let in_left1_poly = MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
            layer.left[1].clone(),
        ));
        let in_right1_poly = MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
            layer.right[1].clone(),
        ));
        // let eq_poly2_1 = MultilinearPolynomial::<F>::from(eq_evals[eq_evals.len() / 2..].to_vec());
        let mut polys2 = vec![eq_poly, in_left1_poly, in_right1_poly];
        let mut workers_polys = vec![&mut polys1, &mut polys2];
        println!("/debug start--------------");
        let (proof, r_sumcheck, final_evals) =
            SumcheckInstanceProof::<F, KeccakTranscript>::simulate_distibuted_prove_arbitrary(
                &running_claim,
                num_rounds,
                &mut workers_polys,
                chunk_size_per_worker,
                |evals| evals[0] * evals[1] * evals[2],
                3,
                &mut transcript,
            );
        println!("-----------------debug end/");
        let left_claim = final_evals[1];
        let right_claim = final_evals[2];

        println!("left_claim: {} right_claim: {}", left_claim, right_claim);
        println!("eq_claim: {}", final_evals[0]);

        layer_proofs.push(LayerProof {
            proof,
            left_claim,
            right_claim,
        });

        let r_layer = transcript.challenge_scalar();
        running_claim = left_claim + r_layer * (right_claim - left_claim);

        println!("layer {} claim: {:?}", layer.layer_idx, running_claim);

        r_grand_product = r_sumcheck.iter().rev().copied().collect();
        r_grand_product.push(r_layer); // pass r_grand_product2 to next layer
        num_rounds += 1;
        chunk_size_per_worker *= 2;
        println!("-------------");
    }

    // Verification

    let mut transcript = KeccakTranscript::new(&[]);
    let mut num_rounds = output_mle.get_num_vars();
    let mut r_grand_product: Vec<_> = transcript.challenge_vector(num_rounds);
    let mut running_claim = output_mle.evaluate(&r_grand_product);

    for (i, layer_proof) in layer_proofs.iter().enumerate() {
        let (sumcheck_claim, r_sumcheck) = layer_proof
            .proof
            .verify(running_claim, num_rounds, 3, &mut transcript)
            .unwrap();

        let eq_eval: F = r_grand_product
            .iter()
            .zip_eq(r_sumcheck.iter().rev())
            .map(|(&r_gp, &r_sc)| r_gp * r_sc + (F::ONE - r_gp) * (F::ONE - r_sc))
            .product();

        println!("eq_eval check {}", eq_eval);

        // assert_eq!(layer1_final_evals[0], layer1_eq_eval); // sanity check

        // verifier check
        assert_eq!(
            layer_proof.left_claim * layer_proof.right_claim * eq_eval,
            sumcheck_claim
        );
        println!("layer {} - verified!", i + 1);

        let r_layer = transcript.challenge_scalar();
        running_claim =
            layer_proof.left_claim + r_layer * (layer_proof.right_claim - layer_proof.left_claim);

        r_grand_product = r_sumcheck.iter().rev().copied().collect();
        r_grand_product.push(r_layer); // pass r_grand_product2 to next layer
        num_rounds += 1;
    }

    // Opennings

    // println!("r_grand_product_inputs: {:?}", r_grand_product_inputs.len());
    // println!("in_left: {:?} in_right: {:?}", in_left, in_right);

    // let in_left_poly = DensePolynomial::new_padded(in_left);
    // let in_right_poly = DensePolynomial::new_padded(in_right);

    // let in_left_eval = in_left_poly.evaluate(&sigma_r_split(&r_grand_product_inputs));
    // let in_right_eval = in_right_poly.evaluate(&sigma_r_split(&r_grand_product_inputs));

    // println!("in_left_eval: {:?}", in_left_eval);
    // println!("in_right_eval: {:?}", in_right_eval);

    // assert_eq!(
    //     layer2_left_claim,
    //     DensePolynomial::new_padded(left_layer2).evaluate(&r_grand_product_inputs)
    // );
    // assert_eq!(
    //     layer2_right_claim,
    //     DensePolynomial::new_padded(right_layer2).evaluate(&r_grand_product_inputs)
    // );
    // assert_eq!(layer2_left_claim, in_left_eval);
    // assert_eq!(layer2_right_claim, in_right_eval);
}

#[test]
fn test_local_gkr_simulation() {
    type F = ark_bn254::Fr;

    let CHUNK_SIZE: usize = env::var("CHUNK_SIZE")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap();
    let N: usize = env::var("BATCH_SIZE")
        .unwrap_or_else(|_| "4".to_string())
        .parse()
        .unwrap();

    println!(
        "CHUNK_SIZE={}/per_worker={}; BATCH_SIZE={}",
        CHUNK_SIZE,
        CHUNK_SIZE / 2,
        N
    );

    let leaves_len = CHUNK_SIZE * N;
    let num_layers = (leaves_len / N).log_2();
    println!("num_layers: {}", num_layers);

    let K = CHUNK_SIZE / 2;
    let K_worker = K / 2;

    let mut in_left = vec![1; K].into_iter().map(F::from).collect::<Vec<_>>();
    let mut in_right = vec![1; K].into_iter().map(F::from).collect::<Vec<_>>();

    for i in 1..N {
        in_left.extend(
            vec![1; K as usize]
                .into_iter()
                .map(|e| F::from(e) + F::from(i as u64))
                .collect::<Vec<_>>(),
        );
        in_right.extend(
            vec![1; K as usize]
                .into_iter()
                .map(|e| F::from(e) + F::from(i as u64))
                .collect::<Vec<_>>(),
        );
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

    println!("\nConstruct layers----------------");

    let mut input_layer = LayerCircuit {
        layer_idx: num_layers,
        left: vec![in_left],
        right: vec![in_right],
    };

    println!(
        "input layer | left: {:?} right: {:?}",
        input_layer.left, input_layer.right
    );

    let mut layers = vec![input_layer];

    // let num_layers = 2;

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

        // println!("pre_last_outputs (sigma): {:?}", pre_last_outputs);

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
    let gkr_output = izip!(last_layer.left[0].clone(), last_layer.right[0].clone())
        .map(|(left, right)| left * right)
        .collect::<Vec<_>>();

    println!("gkr output {:?}", gkr_output);

    println!("\n Prover----------------");

    //------ Output layer (N) prover
    let mut layer_proofs = vec![];
    let mut transcript = KeccakTranscript::new(&[]);

    let output_mle = DensePolynomial::new_padded(gkr_output.clone());
    let mut num_rounds = output_mle.get_num_vars();
    let mut r_grand_product: Vec<_> = transcript.challenge_vector(num_rounds);
    let mut running_claim = output_mle.evaluate(&r_grand_product);

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
            SumcheckInstanceProof::<F, KeccakTranscript>::prove_arbitrary(
                &running_claim,
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
        running_claim = left_claim + r_layer * (right_claim - left_claim);

        println!("layer {} claim: {:?}", layer.layer_idx, running_claim);

        r_grand_product = r_sumcheck.iter().rev().copied().collect();
        r_grand_product.push(r_layer); // pass r_grand_product to next layer
        num_rounds += 1;
    }

    // Verification

    let mut transcript = KeccakTranscript::new(&[]);
    let mut num_rounds = output_mle.get_num_vars();
    let mut r_grand_product: Vec<_> = transcript.challenge_vector(num_rounds);
    let mut running_claim = output_mle.evaluate(&r_grand_product);

    for (i, layer_proof) in layer_proofs.iter().enumerate() {
        let (sumcheck_claim, r_sumcheck) = layer_proof
            .proof
            .verify(running_claim, num_rounds, 3, &mut transcript)
            .unwrap();

        let eq_eval: F = r_grand_product
            .iter()
            .zip_eq(r_sumcheck.iter().rev())
            .map(|(&r_gp, &r_sc)| r_gp * r_sc + (F::ONE - r_gp) * (F::ONE - r_sc))
            .product();

        println!("eq_eval check {}", eq_eval);

        // assert_eq!(layer1_final_evals[0], layer1_eq_eval); // sanity check

        // verifier check
        assert_eq!(
            layer_proof.left_claim * layer_proof.right_claim * eq_eval,
            sumcheck_claim
        );
        println!("layer {} - verified!", i + 1);

        let r_layer = transcript.challenge_scalar();
        running_claim =
            layer_proof.left_claim + r_layer * (layer_proof.right_claim - layer_proof.left_claim);

        r_grand_product = r_sumcheck.iter().rev().copied().collect();
        r_grand_product.push(r_layer); // pass r_grand_product2 to next layer
        num_rounds += 1;
    }
}

#[test]
fn old_test_sumcheck_instance_proof() {
    type F = ark_bn254::Fr;

    let mut rng = ark_std::test_rng();
    let mut in_left = (0u64..8).map(F::from).collect::<Vec<_>>();
    let mut in_right = (8u64..16).map(F::from).collect::<Vec<_>>();

    const N: u64 = 2;

    for i in 1u64..N {
        in_left.extend(
            (0u64..8)
                .map(|e| F::from(e) + F::from(i))
                .collect::<Vec<_>>(),
        );
        in_right.extend(
            (8u64..16)
                .map(|e| F::from(e) + F::from(i))
                .collect::<Vec<_>>(),
        );
    }
    // // Normal scenario
    // {
    //     let layer2 = interleave(&in_left, &in_right).collect_vec();
    //     println!("local layer2: {:?}", layer2);
    //     let layer1 = layer2.chunks(2).map(|c| c[0] * c[1]).collect_vec();
    //     println!("local layer1: {:?}", layer1);
    //     let output = layer1.chunks(2).map(|c| c[0] * c[1]).collect_vec();
    //     println!("local output: {:?}", output);
    //     println!("----------------");
    // }

    let local_input_full = interleave(in_left.clone(), in_right.clone()).collect_vec();
    println!("local input full: {:?}", local_input_full);
    println!("layer2 left_: {:?} layer2 right_: {:?}", in_left, in_right);

    // inputs hold by each worker - cannot be changed!
    let (in_left0, in_left1) = uninterleave_with_padding(&in_left);
    let (in_right0, in_right1) = uninterleave_with_padding(&in_right);

    println!(
        "layer2 left__: {:?} layer2 right__: {:?}",
        [in_left0.clone(), in_left1.clone()],
        [in_right0.clone(), in_right1.clone()]
    );

    // layer2 outputs from each worker - cannot be changed!
    let layer1_coeffs = [
        izip!(&in_left0, &in_right0)
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>(),
        izip!(&in_left1, &in_right1)
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>(),
    ];

    println!("distributed layer1: {:?}", layer1_coeffs);

    // aggregated layer2 outputs from workers - can be changed but final output must be same!
    let layer1_sigma_coeffs =
        interleave(layer1_coeffs[0].clone(), layer1_coeffs[1].clone()).collect_vec();

    println!("aggregated (permuted) layer1: {:?}", layer1_sigma_coeffs);

    let output_coeffs = layer1_sigma_coeffs
        .chunks(2)
        .map(|c| c[0] * c[1])
        .collect_vec();
    println!("aggregated output: {:?}", output_coeffs);

    println!("----------------");

    // let (left_layer2, right_layer2) = (
    //     [in_left0.clone(), in_left1.clone()].concat(),
    //     [in_right0.clone(), in_right1.clone()].concat(),
    // );

    // println!(
    //     "layer2 left: {:?} layer2 right: {:?}",
    //     left_layer2, right_layer2
    // );
    let outputs_layer2 = layer1_coeffs.concat();

    // println!("outputs_layer2: {:?}", outputs_layer2);

    //------ Layer 1 prover

    // println!(
    //     "layer1 left: {:?} layer1 right: {:?}",
    //     layer1_coeffs[0], layer1_coeffs[1]
    // );

    let left_poly = MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
        layer1_coeffs[0].clone(),
    ));
    let right_poly = MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(
        layer1_coeffs[1].clone(),
    ));

    let mut transcript = KeccakTranscript::new(&[]);

    let output_mle = DensePolynomial::new_padded(output_coeffs.clone());
    let r_grand_product: Vec<_> = (0..output_mle.get_num_vars())
        .map(|_| F::rand(&mut rng))
        .collect();
    let eq_poly = MultilinearPolynomial::<F>::from(EqPolynomial::evals(&r_grand_product));
    // println!("eq_poly: {:?}", eq_poly.coeffs_as_field_elements());
    let output_claim = output_mle.evaluate(&r_grand_product);
    let num_rounds = r_grand_product.len();
    // println!("Layer1 rounds: {}", num_rounds);

    let mut polys = vec![eq_poly, left_poly, right_poly];
    let (layer1_proof, layer1_r_sumcheck, layer1_final_evals) =
        SumcheckInstanceProof::<F, KeccakTranscript>::prove_arbitrary(
            &output_claim,
            num_rounds,
            &mut polys,
            |evals| evals[0] * evals[1] * evals[2],
            3,
            &mut transcript,
        );

    // println!("layer1_final_evals: {:?}", layer1_final_evals);

    let layer_1_left_claim = layer1_final_evals[1];
    let layer1_right_claim = layer1_final_evals[2];

    let r_layer = F::rand(&mut rng);
    let layer2_claim = layer_1_left_claim + r_layer * (layer1_right_claim - layer_1_left_claim);

    // println!("layer2_claim: {:?}", layer2_claim);

    let mut r_grand_product2: Vec<F> = layer1_r_sumcheck.iter().rev().copied().collect();
    r_grand_product2.push(r_layer); // pass r_grand_product2 to next layer

    // println!("layer1 - proved!");

    //------ Layer 2 prover

    println!("Layer2 rounds: {}", num_rounds + 1);

    // println!("r_grand_product2: {:?}", r_grand_product2);

    let layer2_output_mle = DensePolynomial::new_padded(outputs_layer2.clone());
    // let eq_poly2 =
    //     MultilinearPolynomial::<F>::from(EqPolynomial::evals(&sigma_r_split(&r_grand_product2)));

    // println!("layer2_output_mle: {:?}", layer2_output_mle.Z);

    // println!("eq_poly2: {:?}", eq_poly2.coeffs_as_field_elements());

    {
        println!("local sumcheck--------------");
        let eq_poly2 = MultilinearPolynomial::<F>::from(EqPolynomial::evals(&r_grand_product2));
        let mut transcript = transcript.clone();
        let left_poly2 =
            MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(in_left.clone()));
        let right_poly2 =
            MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(in_right.clone()));
        let mut polys = vec![eq_poly2.clone(), left_poly2, right_poly2];

        let (layer2_proof, layer2_r_sumcheck, layer2_final_evals) =
            SumcheckInstanceProof::<F, KeccakTranscript>::prove_arbitrary(
                &layer2_claim,
                num_rounds + 1,
                &mut polys,
                |evals| evals[0] * evals[1] * evals[2],
                3,
                &mut transcript,
            );
        println!(
            "check sumcheck done with final evals {:?}",
            layer2_final_evals
        );
        println!("---------------------------");
    }

    println!("distributed sumcheck--------------");

    let mut in_left0 = (0u64..4).map(F::from).collect::<Vec<_>>();
    let mut in_left1 = (4u64..8).map(F::from).collect::<Vec<_>>();
    let mut in_right0 = (8u64..12).map(F::from).collect::<Vec<_>>();
    let mut in_right1 = (12u64..16).map(F::from).collect::<Vec<_>>();

    for i in 1u64..N {
        in_left0.extend(
            (0u64..4)
                .map(|e| F::from(e) + F::from(i))
                .collect::<Vec<_>>(),
        );
        in_left1.extend(
            (4u64..8)
                .map(|e| F::from(e) + F::from(i))
                .collect::<Vec<_>>(),
        );

        in_right0.extend(
            (8u64..12)
                .map(|e| F::from(e) + F::from(i))
                .collect::<Vec<_>>(),
        );
        in_right1.extend(
            (12u64..16)
                .map(|e| F::from(e) + F::from(i))
                .collect::<Vec<_>>(),
        );
    }

    println!("in_left: {:?}", [in_left0.clone(), in_left1.clone()]);
    println!("in_right: {:?}", [in_right0.clone(), in_right1.clone()]);
    // let eq_evals = EqPolynomial::evals(&r_grand_product2);
    let eq_poly2 = MultilinearPolynomial::<F>::from(EqPolynomial::evals(&r_grand_product2));

    let in_left0_poly2 =
        MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(in_left0.clone()));
    let in_right0_poly2 =
        MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(in_right0.clone()));
    // let eq_poly2_0 = MultilinearPolynomial::<F>::from(eq_evals[0..eq_evals.len() / 2].to_vec());
    let mut polys1 = vec![eq_poly2.clone(), in_left0_poly2, in_right0_poly2];
    let in_left1_poly2 =
        MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(in_left1.clone()));
    let in_right1_poly2 =
        MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new_padded(in_right1.clone()));
    // let eq_poly2_1 = MultilinearPolynomial::<F>::from(eq_evals[eq_evals.len() / 2..].to_vec());
    let mut polys2 = vec![eq_poly2, in_left1_poly2, in_right1_poly2];
    let mut polys = vec![&mut polys1, &mut polys2];
    let (layer2_proof, layer2_r_sumcheck, layer2_final_evals) =
        SumcheckInstanceProof::<F, KeccakTranscript>::simulate_distibuted_prove_arbitrary(
            &layer2_claim,
            num_rounds + 1,
            &mut polys,
            4,
            |evals| evals[0] * evals[1] * evals[2],
            3,
            &mut transcript,
        );
    println!("---------------------------");

    let layer2_left_claim = layer2_final_evals[1];
    let layer2_right_claim = layer2_final_evals[2];

    let r_layer2 = F::rand(&mut rng);
    let gkr_inputs_claim = layer2_left_claim + r_layer2 * (layer2_right_claim - layer2_left_claim);

    println!("layer2_claim: {:?}", layer2_claim);

    let mut r_grand_product_final: Vec<F> = layer2_r_sumcheck.iter().rev().copied().collect();
    // r_grand_product_final.push(r_layer2); // pass r_grand_product2 to next layer

    let r_grand_product_inputs = r_grand_product_final;

    println!("----------------");

    // Verification

    let mut transcript = KeccakTranscript::new(&[]);
    let (sumcheck_claim, r_sumcheck) = layer1_proof
        .verify(output_claim, num_rounds, 3, &mut transcript)
        .unwrap();

    let layer1_eq_eval: F = r_grand_product
        .iter()
        .zip_eq(r_sumcheck.iter().rev())
        .map(|(&r_gp, &r_sc)| r_gp * r_sc + (F::ONE - r_gp) * (F::ONE - r_sc))
        .product();

    assert_eq!(layer1_final_evals[0], layer1_eq_eval); // sanity check

    // verifier check
    assert_eq!(
        layer_1_left_claim * layer1_right_claim * layer1_eq_eval,
        sumcheck_claim
    );
    println!("layer1 - verified!");

    let (sumcheck_claim, r_sumcheck) = layer2_proof
        .verify(layer2_claim, num_rounds + 1, 3, &mut transcript)
        .unwrap();

    // let r_grand_product2 = sigma_r_split(&r_grand_product2);

    let layer2_eq_eval: F = r_grand_product2
        .iter()
        .zip_eq(r_sumcheck.iter().rev())
        .map(|(&r_gp, &r_sc)| r_gp * r_sc + (F::ONE - r_gp) * (F::ONE - r_sc))
        .product();

    assert_eq!(layer2_final_evals[0], layer2_eq_eval); // sanity check

    assert_eq!(
        layer2_left_claim * layer2_right_claim * layer2_eq_eval,
        sumcheck_claim
    );
    println!("layer2 - verified!");
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

fn uninterleave<T: Clone>(v: &[T]) -> (Vec<T>, Vec<T>) {
    (
        v.iter().cloned().step_by(2).collect(),
        v.iter().cloned().skip(1).step_by(2).collect(),
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
