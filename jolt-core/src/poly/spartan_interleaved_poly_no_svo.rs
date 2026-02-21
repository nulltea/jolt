#![allow(clippy::too_many_arguments)]

use crate::field::JoltField;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::utils::math::Math;
use crate::zkvm::r1cs::constraints::UNIFORM_R1CS;
use crate::zkvm::r1cs::inputs::R1CSCycleInputs;
use crate::zkvm::JoltSharedPreprocessing;
use rayon::prelude::*;
use tracer::instruction::Cycle;

/// Sparse interleaved representation of the Stage 1 Spartan outer sumcheck polynomials (no SVO).
///
/// We represent the three multilinear polynomials `Az`, `Bz`, `Cz` over the row index `x`
/// interleaved in a single sparse vector:
/// - row `k` has coefficients at indices `3k+0`, `3k+1`, `3k+2` (Az/Bz/Cz).
///
/// When binding variables in **LowToHigh** order, sumcheck rounds pair adjacent rows in `k`,
/// so a block of 6 entries (two rows × 3 polys) shares the same `index / 6`.
#[derive(Clone, Debug)]
pub struct SpartanInterleavedPolynomialNoSvo<F: JoltField> {
    pub(crate) unbound_coeffs_shards: Vec<Vec<SparseCoefficient<F>>>,
    pub(crate) bound_coeffs: Vec<SparseCoefficient<F>>,
    dense_len: usize,
    padded_num_constraints: usize,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseCoefficient<T> {
    pub index: usize,
    pub value: T,
}

impl<T> From<(usize, T)> for SparseCoefficient<T> {
    fn from(x: (usize, T)) -> Self {
        Self {
            index: x.0,
            value: x.1,
        }
    }
}

impl<F: JoltField> SpartanInterleavedPolynomialNoSvo<F> {
    #[tracing::instrument(skip_all, name = "SpartanInterleavedPolynomialNoSvo::new")]
    pub fn new(preprocessing: &JoltSharedPreprocessing, trace: &[Cycle]) -> Self {
        let padded_num_constraints = UNIFORM_R1CS.len().next_power_of_two();
        let num_steps = trace.len();
        let dense_len = num_steps * padded_num_constraints;
        debug_assert!(dense_len.is_power_of_two());

        let num_chunks = core::cmp::min(
            rayon::current_num_threads().next_power_of_two() * 16,
            core::cmp::max(1, num_steps / 2),
        );
        let chunk_size = num_steps.div_ceil(num_chunks);

        let shards: Vec<Vec<SparseCoefficient<F>>> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = core::cmp::min((chunk_idx + 1) * chunk_size, num_steps);
                let mut coeffs: Vec<SparseCoefficient<F>> =
                    Vec::with_capacity((end - start) * UNIFORM_R1CS.len() * 3);

                for step_idx in start..end {
                    let row_inputs =
                        R1CSCycleInputs::from_trace::<F>(preprocessing, trace, step_idx);
                    for (constraint_idx, named) in UNIFORM_R1CS.iter().enumerate() {
                        let row_index = step_idx * padded_num_constraints + constraint_idx;
                        let base = 3 * row_index;

                        let az = named.cons.a.evaluate_row_with::<F>(&row_inputs);
                        if !az.is_zero() {
                            coeffs.push((base, az).into());
                        }

                        let bz = named.cons.b.evaluate_row_with::<F>(&row_inputs);
                        if !bz.is_zero() {
                            coeffs.push((base + 1, bz).into());
                        }

                        let cz = named.cons.c.evaluate_row_with::<F>(&row_inputs);
                        if !cz.is_zero() {
                            coeffs.push((base + 2, cz).into());
                        }
                    }
                }

                coeffs
            })
            .collect();

        Self {
            unbound_coeffs_shards: shards,
            bound_coeffs: vec![],
            dense_len,
            padded_num_constraints,
        }
    }

    pub fn is_bound(&self) -> bool {
        !self.bound_coeffs.is_empty()
    }

    #[tracing::instrument(
        skip_all,
        name = "SpartanInterleavedPolynomialNoSvo::first_sumcheck_round",
        level = "trace"
    )]
    pub fn first_sumcheck_round<ProofTranscript: crate::transcripts::Transcript>(
        &mut self,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r: &mut Vec<F::Challenge>,
        polys: &mut Vec<crate::poly::unipoly::CompressedUniPoly<F>>,
        claim: &mut F,
    ) {
        debug_assert!(!self.is_bound());

        let (t0, t_inf) = quadratic_evals_from_unbound(
            &self.unbound_coeffs_shards,
            eq_poly,
            self.padded_num_constraints,
            self.dense_len,
        );

        let r_i = crate::subprotocols::sumcheck::process_eq_sumcheck_round(
            (t0, t_inf),
            eq_poly,
            polys,
            r,
            claim,
            transcript,
        );

        self.bound_coeffs = bind_sparse_shards_into_bound(&self.unbound_coeffs_shards, r_i.into());
        self.unbound_coeffs_shards.clear();
        self.unbound_coeffs_shards.shrink_to_fit();
        self.dense_len /= 2;
    }

    #[tracing::instrument(
        skip_all,
        name = "SpartanInterleavedPolynomialNoSvo::subsequent_sumcheck_round",
        level = "trace"
    )]
    pub fn subsequent_sumcheck_round<ProofTranscript: crate::transcripts::Transcript>(
        &mut self,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r: &mut Vec<F::Challenge>,
        polys: &mut Vec<crate::poly::unipoly::CompressedUniPoly<F>>,
        claim: &mut F,
    ) {
        debug_assert!(self.is_bound());

        let (t0, t_inf) = quadratic_evals_from_bound(&self.bound_coeffs, eq_poly, self.dense_len);

        let r_i = crate::subprotocols::sumcheck::process_eq_sumcheck_round(
            (t0, t_inf),
            eq_poly,
            polys,
            r,
            claim,
            transcript,
        );

        self.bound_coeffs = bind_sparse_coeffs_low_to_high(&self.bound_coeffs, r_i.into());
        self.dense_len /= 2;
    }

    pub fn final_sumcheck_evals(&self) -> [F; 3] {
        debug_assert_eq!(self.dense_len, 1);
        let mut out = [F::zero(), F::zero(), F::zero()];
        for coeff in self.bound_coeffs.iter() {
            out[coeff.index % 3] = coeff.value;
        }
        out
    }
}

fn quadratic_evals_from_unbound<F: JoltField>(
    shards: &[Vec<SparseCoefficient<F>>],
    eq_poly: &GruenSplitEqPolynomial<F>,
    padded_num_constraints: usize,
    dense_len: usize,
) -> (F, F) {
    let _ = padded_num_constraints;
    let num_blocks = dense_len / 2;

    shards
        .par_iter()
        .map(|coeffs| {
            let mut t0 = F::zero();
            let mut t_inf = F::zero();

            let mut i = 0;
            while i < coeffs.len() {
                let block = coeffs[i].index / 6;

                let mut az0 = F::zero();
                let mut bz0 = F::zero();
                let mut cz0 = F::zero();
                let mut az1 = F::zero();
                let mut bz1 = F::zero();
                let mut _cz1 = F::zero();

                while i < coeffs.len() && coeffs[i].index / 6 == block {
                    match coeffs[i].index % 6 {
                        0 => az0 = coeffs[i].value,
                        1 => bz0 = coeffs[i].value,
                        2 => cz0 = coeffs[i].value,
                        3 => az1 = coeffs[i].value,
                        4 => bz1 = coeffs[i].value,
                        5 => _cz1 = coeffs[i].value,
                        _ => unreachable!(),
                    }
                    i += 1;
                }

                let weight = weight_for_block(eq_poly, block, num_blocks);

                let az_inf = az1 - az0;
                let bz_inf = bz1 - bz0;

                let abc0 = az0 * bz0 - cz0;
                let ab_inf = az_inf * bz_inf;

                t0 += abc0 * weight;
                t_inf += ab_inf * weight;
            }

            (t0, t_inf)
        })
        .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
}

fn quadratic_evals_from_bound<F: JoltField>(
    coeffs: &[SparseCoefficient<F>],
    eq_poly: &GruenSplitEqPolynomial<F>,
    dense_len: usize,
) -> (F, F) {
    let num_blocks = dense_len / 2;

    let mut t0 = F::zero();
    let mut t_inf = F::zero();

    let mut i = 0;
    while i < coeffs.len() {
        let block = coeffs[i].index / 6;

        let mut az0 = F::zero();
        let mut bz0 = F::zero();
        let mut cz0 = F::zero();
        let mut az1 = F::zero();
        let mut bz1 = F::zero();
        let mut _cz1 = F::zero();

        while i < coeffs.len() && coeffs[i].index / 6 == block {
            match coeffs[i].index % 6 {
                0 => az0 = coeffs[i].value,
                1 => bz0 = coeffs[i].value,
                2 => cz0 = coeffs[i].value,
                3 => az1 = coeffs[i].value,
                4 => bz1 = coeffs[i].value,
                5 => _cz1 = coeffs[i].value,
                _ => unreachable!(),
            }
            i += 1;
        }

        let weight = weight_for_block(eq_poly, block, num_blocks);

        let az_inf = az1 - az0;
        let bz_inf = bz1 - bz0;

        let abc0 = az0 * bz0 - cz0;
        let ab_inf = az_inf * bz_inf;

        t0 += abc0 * weight;
        t_inf += ab_inf * weight;
    }

    (t0, t_inf)
}

#[inline(always)]
fn weight_for_block<F: JoltField>(
    eq_poly: &GruenSplitEqPolynomial<F>,
    block: usize,
    num_blocks: usize,
) -> F {
    let e_in_len = eq_poly.E_in_current_len().max(1);
    let e_out_len = eq_poly.E_out_current_len().max(1);
    let covered = e_in_len * e_out_len;
    debug_assert!(covered > 0);
    debug_assert!(num_blocks % covered == 0);
    let collapse = num_blocks / covered;
    let block_collapsed = if collapse > 1 {
        block / collapse
    } else {
        block
    };

    if e_in_len <= 1 {
        eq_poly.E_out_current()[block_collapsed]
    } else {
        let num_x_in_bits = e_in_len.log_2();
        let x_in_mask = (1usize << num_x_in_bits) - 1;
        let x_in = block_collapsed & x_in_mask;
        let x_out = block_collapsed >> num_x_in_bits;
        eq_poly.E_in_current()[x_in] * eq_poly.E_out_current()[x_out]
    }
}

fn bind_sparse_shards_into_bound<F: JoltField>(
    shards: &[Vec<SparseCoefficient<F>>],
    r: F,
) -> Vec<SparseCoefficient<F>> {
    let output_lens: Vec<usize> = shards
        .par_iter()
        .map(|coeffs| binding_output_length(coeffs))
        .collect();
    let total_len: usize = output_lens.iter().sum();
    let mut out: Vec<SparseCoefficient<F>> = Vec::with_capacity(total_len);

    for (coeffs, expected_len) in shards.iter().zip(output_lens.into_iter()) {
        let before = out.len();
        out.extend(bind_sparse_coeffs_low_to_high(coeffs, r));
        debug_assert_eq!(out.len() - before, expected_len);
    }

    out
}

fn binding_output_length<F: JoltField>(coeffs: &[SparseCoefficient<F>]) -> usize {
    let mut out = 0usize;
    let mut i = 0;
    while i < coeffs.len() {
        let block = coeffs[i].index / 6;
        let mut has_a = false;
        let mut has_b = false;
        let mut has_c = false;
        while i < coeffs.len() && coeffs[i].index / 6 == block {
            match coeffs[i].index % 6 {
                0 | 3 => has_a = true,
                1 | 4 => has_b = true,
                2 | 5 => has_c = true,
                _ => unreachable!(),
            }
            i += 1;
        }
        out += (has_a as usize) + (has_b as usize) + (has_c as usize);
    }
    out
}

fn bind_sparse_coeffs_low_to_high<F: JoltField>(
    coeffs: &[SparseCoefficient<F>],
    r: F,
) -> Vec<SparseCoefficient<F>> {
    let mut out: Vec<SparseCoefficient<F>> = Vec::with_capacity(binding_output_length(coeffs));
    let mut i = 0;
    while i < coeffs.len() {
        let block = coeffs[i].index / 6;

        let mut a0 = None;
        let mut b0 = None;
        let mut c0 = None;
        let mut a1 = None;
        let mut b1 = None;
        let mut c1 = None;

        while i < coeffs.len() && coeffs[i].index / 6 == block {
            match coeffs[i].index % 6 {
                0 => a0 = Some(coeffs[i].value),
                1 => b0 = Some(coeffs[i].value),
                2 => c0 = Some(coeffs[i].value),
                3 => a1 = Some(coeffs[i].value),
                4 => b1 = Some(coeffs[i].value),
                5 => c1 = Some(coeffs[i].value),
                _ => unreachable!(),
            }
            i += 1;
        }

        let out_base = 3 * block;

        if a0.is_some() || a1.is_some() {
            let (low, high) = (a0.unwrap_or(F::zero()), a1.unwrap_or(F::zero()));
            let v = low + (high - low) * r;
            if !v.is_zero() {
                out.push((out_base, v).into());
            }
        }

        if b0.is_some() || b1.is_some() {
            let (low, high) = (b0.unwrap_or(F::zero()), b1.unwrap_or(F::zero()));
            let v = low + (high - low) * r;
            if !v.is_zero() {
                out.push((out_base + 1, v).into());
            }
        }

        if c0.is_some() || c1.is_some() {
            let (low, high) = (c0.unwrap_or(F::zero()), c1.unwrap_or(F::zero()));
            let v = low + (high - low) * r;
            if !v.is_zero() {
                out.push((out_base + 2, v).into());
            }
        }
    }

    out
}
