//! Implements the Dao-Thaler optimization for EQ polynomial evaluations
//! https://eprint.iacr.org/2024/1210.pdf
use std::env;

use itertools::Itertools;

// #[cfg(test)]
use super::dense_mlpoly::DensePolynomial;
use crate::{field::JoltField, poly::eq_poly::EqPolynomial, utils::math::Math};

#[derive(Debug, Clone, PartialEq)]
/// A struct holding the equality polynomial evaluations for use in sum-check, when incorporating
/// both the Gruen and Dao-Thaler optimizations.
///
/// For the `i = 0..n`-th round of sum-check, we want the following invariants:
///
/// - `current_index = n - i` (where `n = w.len()`)
/// - `current_scalar = eq(w[(n - i)..],r[..i])`
/// - `E_out_vec.last().unwrap() = [eq(w[..min(i, n/2)], x) for all x in {0, 1}^{n - min(i, n/2)}]`
/// - If `i < n/2`, then `E_in_vec.last().unwrap() = [eq(w[n/2..(n/2 + i + 1)], x) for all x in {0,
///   1}^{n/2 - i - 1}]`; else `E_in_vec` is empty
///
/// Note: all current applications of `SplitEqPolynomial` use the `LowToHigh` binding order. This
/// means that we are iterating over `w` in the reverse order: `w.len()` down to `0`.
pub struct GruenSplitEqPolynomial<F> {
    pub current_index: usize,
    pub current_scalar: F,
    pub w: Vec<F>,
    pub E_in_vec: Vec<Vec<F>>,
    pub E_out_vec: Vec<Vec<F>>,
}

/// Old struct for split equality polynomial, without Gruen's optimization
/// TODO: remove all usage of this struct with the new one
pub struct SplitEqPolynomial<F> {
    pub num_vars: usize,
    pub E1: Vec<F>,
    pub E1_len: usize,
    pub E2: Vec<F>,
    pub E2_len: usize,
}

impl<F: JoltField> GruenSplitEqPolynomial<F> {
    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::new")]
    pub fn new(w: &[F]) -> Self {
        let m = w.len() / 2;
        //   w = [w_out, w_in, w_last]
        //         ↑      ↑      ↑
        //         |      |      |
        //         |      |      last element
        //         |      second half of remaining elements (for E_in)
        //         first half of remaining elements (for E_out)
        let (_, wprime) = w.split_last().unwrap();
        let (w_out, w_in) = wprime.split_at(m);
        let (E_out_vec, E_in_vec) = rayon::join(
            || EqPolynomial::evals_cached(w_out),
            || EqPolynomial::evals_cached(w_in),
        );
        Self {
            current_index: w.len(),
            current_scalar: F::one(),
            w: w.to_vec(),
            E_in_vec,
            E_out_vec,
        }
    }

    /// Compute the split equality polynomial for the small value optimization
    ///
    /// The split is done as follows: (here `l = num_small_value_rounds`)
    ///
    /// 0 ..... (n/2 - l) ..... (n - l) ..... n
    ///
    ///           <-- E_in -->
    ///
    /// E_out --->                <--- E_out
    ///
    /// where the first E_out part (0 to n/2 - l) corresponds to x_out, and the second E_out part
    /// (n/2 - l to n) corresponds to y_suffix
    ///
    /// Returns E_out which contains `l` vectors of eq evals for the same x_out part, with decreasing
    /// length for y_suffix, and E_in which contains the single vector of eq evals for the x_in part.
    ///
    /// Note the differences between this and the `new` constructor: this is specialized for the
    /// small value optimization.
    pub fn new_for_small_value(
        w: &[F],
        num_x_out_vars: usize,
        num_x_in_vars: usize,
        num_small_value_rounds: usize,
    ) -> Self {
        // Split w into the slices: (l = num_small_value_rounds)
        // (n/2 - l) ..... (n - l)
        // 0..(n/2 - l - 1) concatenated with (n - l)...n
        // Then invoke the evals_cached constructor on the concatenated slice, producing E_out
        // Invoke the evals constructor (no caching) on the middle slice, producing E_in
        // In other words, there is only 1 vector in E_in, and l vectors in E_out
        // (we may drop the rest of the vectors after evals_cached)
        let n = w.len();

        assert!(
            n > 0,
            "length of w must be positive for the split to be valid."
        );
        assert!(num_x_out_vars + num_x_in_vars + num_small_value_rounds == n, "num_x_out_vars + num_x_in_vars + num_small_value_rounds must be == n for the split to be valid.");

        // This should be `min(num_steps, n/2 - num_small_value_rounds)`, computed externally before calling this function.
        let split_point_x_out = num_x_out_vars;
        let split_point_x_in = split_point_x_out + num_x_in_vars;

        let w_E_in_vars: Vec<F> = w[split_point_x_out..split_point_x_in].to_vec();

        // Determine the end index for the suffix part of w_E_out_vars
        let suffix_slice_end = if num_small_value_rounds == 0 {
            split_point_x_in // Results in an empty suffix, e.g., w[n..n]
        } else {
            n - 1 // Use up to n-1, excluding the last variable of w (tau)
        };

        let num_actual_suffix_vars = suffix_slice_end.saturating_sub(split_point_x_in);

        let mut w_E_out_vars: Vec<F> = Vec::with_capacity(num_x_out_vars + num_actual_suffix_vars);
        w_E_out_vars.extend_from_slice(&w[0..split_point_x_out]);
        if split_point_x_in < suffix_slice_end {
            // Add suffix only if range is valid and non-empty
            w_E_out_vars.extend_from_slice(&w[split_point_x_in..suffix_slice_end]);
        }

        let (mut E_out_vec, E_in) = rayon::join(
            || EqPolynomial::evals_cached(&w_E_out_vars),
            || EqPolynomial::evals(&w_E_in_vars),
        );

        // Take only the first `num_small_value_rounds` vectors from E_out_vec (after reversing)
        // Recall that at this point, E_out_vec[0] = `eq(w[0..split_point_x_out] ++ w[split_point_x_in..n-1], x)`
        E_out_vec.reverse();
        E_out_vec.truncate(num_small_value_rounds);

        Self {
            current_index: num_x_out_vars,
            current_scalar: F::one(),
            w: w.to_vec(),
            E_in_vec: vec![E_in],
            E_out_vec,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.w.len()
    }

    pub fn len(&self) -> usize {
        1 << self.current_index
    }

    pub fn E_in_current_len(&self) -> usize {
        self.E_in_vec.last().unwrap().len()
    }

    pub fn E_out_current_len(&self) -> usize {
        self.E_out_vec.last().unwrap().len()
    }

    /// Return the last vector from `E1` as a slice
    pub fn E_in_current(&self) -> &[F] {
        self.E_in_vec.last().unwrap()
    }

    /// Return the last vector from `E2` as a slice
    pub fn E_out_current(&self) -> &[F] {
        self.E_out_vec.last().unwrap()
    }

    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::bind", level = "trace")]
    pub fn bind(&mut self, r: F) {
        // multiply `current_scalar` by `eq(w[i], r) = (1 - w[i]) * (1 - r) + w[i] * r`
        // which is the same as `1 - w[i] - r + 2 * w[i] * r`
        let prod_w_r = self.w[self.current_index - 1] * r;
        self.current_scalar *= F::one() - self.w[self.current_index - 1] - r + prod_w_r + prod_w_r;
        // decrement `current_index`
        self.current_index -= 1;
        // pop the last vector from `E_in_vec` or `E_out_vec` (since we don't need it anymore)
        if self.w.len() / 2 < self.current_index {
            self.E_in_vec.pop();
        } else if 0 < self.current_index {
            self.E_out_vec.pop();
        }
    }

    #[cfg(test)]
    fn to_E1_old(&self) -> Vec<F> {
        if self.current_index > self.w.len() / 2 {
            let wi = self.w[self.current_index - 1];
            let E1_old_odd: Vec<F> = self
                .E_in_vec
                .last()
                .unwrap()
                .iter()
                .map(|x| *x * (F::one() - wi))
                .collect();
            let E1_old_even: Vec<F> = self
                .E_in_vec
                .last()
                .unwrap()
                .iter()
                .map(|x| *x * wi)
                .collect();
            // Interleave the two vectors
            let mut E1_old = vec![];
            for i in 0..E1_old_odd.len() {
                E1_old.push(E1_old_odd[i]);
                E1_old.push(E1_old_even[i]);
            }
            E1_old
        } else {
            // println!("Don't expect to call this");
            vec![self.current_scalar; 1]
        }
    }

    #[cfg(test)]
    pub fn merge(&self) -> DensePolynomial<F> {
        let evals = EqPolynomial::evals(&self.w[..self.current_index])
            .iter()
            .map(|x| *x * self.current_scalar)
            .collect();
        DensePolynomial::new(evals)
    }
}

impl<F: JoltField> SplitEqPolynomial<F> {
    #[tracing::instrument(skip_all, name = "SplitEqPolynomial::new", level = "trace")]
    pub fn new(w: &[F]) -> Self {
        let m = w.len() / 2;
        let (w2, w1) = w.split_at(m);
        let (E2, E1) = rayon::join(|| EqPolynomial::evals(w2), || EqPolynomial::evals(w1));
        let E1_len = E1.len();
        let E2_len = E2.len();
        Self {
            num_vars: w.len(),
            E1,
            E1_len,
            E2,
            E2_len,
        }
    }

    pub fn new_bound(E1: Vec<F>, E2: Vec<F>) -> Self {
        Self {
            num_vars: (E1.len() * E2.len()).log_2(),
            E1_len: E1.len(),
            E2_len: E2.len(),
            E1,
            E2,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        if self.E1_len == 1 {
            self.E2_len
        } else {
            self.E1_len * self.E2_len
        }
    }

    #[tracing::instrument(skip_all, name = "SplitEqPolynomial::bind", level = "trace")]
    pub fn bind(&mut self, r: F) {
        if self.E1_len == 1 {
            // E_1 is already completely bound, so we bind E_2
            let n = self.E2_len / 2;
            for i in 0..n {
                self.E2[i] = self.E2[2 * i] + r * (self.E2[2 * i + 1] - self.E2[2 * i]);
            }
            self.E2_len = n;
        } else {
            // Bind E_1
            let n = self.E1_len / 2;
            for i in 0..n {
                self.E1[i] = self.E1[2 * i] + r * (self.E1[2 * i + 1] - self.E1[2 * i]);
            }
            self.E1_len = n;

            // If E_1 is now completely bound, we will be switching over to the
            // linear-time sumcheck prover, using E_1 * E_2:
            if self.E1_len == 1 {
                self.E2[..self.E2_len]
                    .iter_mut()
                    .for_each(|eval| *eval *= self.E1[0]);
            }
        }
    }

    // #[cfg(test)]
    pub fn merge(&self) -> DensePolynomial<F> {
        if self.E1_len == 1 {
            DensePolynomial::new_padded(self.E2[..self.E2_len].to_vec())
        } else {
            let mut merged = vec![];
            for i in 0..self.E2_len {
                for j in 0..self.E1_len {
                    merged.push(self.E2[i] * self.E1[j])
                }
            }
            DensePolynomial::new_padded(merged)
        }
    }
}

/// A SplitEqPolynomial chunk assigned to a single worker, with:
/// - Dao–Thaler factorization Eq(i_A, i_C, i_B) = E2(i_A, i_B) * E1(i_C),
/// - a *contiguous* 1D slice of the global Eq table [global_start, global_end),
/// - plus metadata to map between:
///     - global EQ indices,
///     - (row, col) indices in the factored table,
///     - local slice indices used to align with DenseInterleavedPolynomial.
pub struct DistributedSplitEqPolynomial<F> {
    /// Number of currently unbound variables *seen by this worker* (A|C),
    /// i.e. total Eq variables minus already-bound ones and minus chunk bits B.
    pub num_vars: usize,

    // -------- factored Dao–Thaler structure over A|C|B --------
    //
    // Semantics match SplitEqPolynomial:
    //
    //   Eq_rect(i_A, i_C, i_B) = E2(i_A, i_B) * E1(i_C)
    //
    // where:
    //   - i_C indexes the "inner" variables C (columns, bound first),
    //   - (i_A, i_B) indexes the "outer" variables A and chunk bits B (rows).
    pub E1: Vec<F>,
    /// Current number of columns in the Dao–Thaler factorization: 2^{|C|} or 1.
    pub E1_len: usize,

    pub E2: Vec<F>,
    /// Current number of rows in the Dao–Thaler factorization: 2^{|A|+|B|_active}.
    pub E2_len: usize,

    /// Global row index of E2[0]. I.e. E2[row_offset] corresponds to the global row
    /// with index row_start + row_offset in the full Eq table.
    pub row_start: usize,

    /// First global Eq index assigned to this worker (flattened row-major index into the
    /// *full* Eq table before Dao–Thaler factorization and chunking).
    pub global_start: usize,

    /// One-past last global Eq index assigned to this worker.
    pub global_end: usize,

    /// Length of this worker’s Eq slice in *points*:
    ///   worker_len = global_end - global_start
    ///
    /// This is the number of Eq points this worker logically owns, even if after binding
    /// the attached polynomial P only covers a prefix of them.
    pub worker_len: usize,
}

impl<F: JoltField> DistributedSplitEqPolynomial<F> {
    #[tracing::instrument(skip_all, name = "DistributedSplitEqPolynomial::new", level = "trace")]
    pub fn new(w: &[F], log_chunks: usize, k: usize, eq_pairs: usize) -> Self {
        // Build the *global* SplitEqPolynomial over all variables A|C|B.
        let base = SplitEqPolynomial::new(w);

        let n_workers = 1usize << log_chunks;
        assert!(k < n_workers, "worker index {} out of {}", k, n_workers);

        // E1_len = number of columns = 2^{|C|}.
        // E2_len = number of rows    = 2^{|A|+|B|}.
        let E1_len_global = base.E1_len;
        let E2_len_global = base.E2_len;
        let total_points = E1_len_global * E2_len_global;

        // The coordinator assigns each worker a *contiguous* 1D slice of the flattened Eq
        // table using [global_start, global_end), measured in Eq points (not rows).
        let global_start = k * eq_pairs;
        assert!(
            global_start < total_points,
            "worker {} starts past end of eq table (start={}, total={})",
            k,
            global_start,
            total_points
        );

        // All non-last workers get exactly `eq_pairs` points, last worker runs to the end.
        let global_end = if k + 1 == n_workers {
            total_points
        } else {
            core::cmp::min(global_start + eq_pairs, total_points)
        };
        assert!(global_end > global_start);

        // Compute the minimal *row interval* [row_start, row_end) in the global Eq
        // table that covers this 1D slice [global_start, global_end). Rows are
        // indexed in row-major order with row width E1_len_global.
        //
        //   row_start = floor(global_start / E1_len)
        //   row_end   = ceil(global_end  / E1_len)
        //
        let row_start = global_start / E1_len_global;
        let mut row_end = (global_end + E1_len_global - 1) / E1_len_global; // ceil
        if row_end > E2_len_global {
            row_end = E2_len_global;
        }
        assert!(row_start < row_end);
        assert!(row_end <= E2_len_global);

        // Restrict E2 to the rows actually needed for this worker.
        let e2_start = row_start;
        let e2_end = row_end;
        let e2_len = e2_end - e2_start;
        let E2 = base.E2[e2_start..e2_end].to_vec();

        // Worker’s logical Eq slice length in points.
        let worker_len = global_end - global_start;

        Self {
            num_vars: w.len() - log_chunks,
            E1: base.E1,
            E1_len: base.E1_len,
            E2,
            E2_len: e2_len,
            row_start,
            global_start,
            global_end,
            worker_len,
        }
    }

    /// Current number of unbound variables (A|C) in this worker’s view.
    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    #[inline]
    pub fn len(&self) -> usize {
        // Number of Eq points in this worker's contiguous slice
        // [global_start, global_end), after any bindings.
        self.worker_len
    }

    /// Bind one sumcheck variable (same order/convention as `SplitEqPolynomial::bind`).
    ///
    /// Semantics:
    /// - If C is non-empty (E1_len > 1), we bind a C-variable:
    ///     - E1_len halves, E1 entries are linearly combined by r,
    ///     - if E1_len becomes 1, we collapse into linear-time mode by scaling E2.
    /// - If C is empty (E1_len == 1), we bind an A|B-variable:
    ///     - E2_len halves, each new row is a combination of two old rows,
    ///     - row_start halves because rows are merged pairwise.
    ///
    /// In all cases, the *global* Eq table halves in length and each new global index
    /// corresponds to floor(old_index / 2), so we also remap [global_start, global_end)
    /// and `worker_len` accordingly.
    pub fn bind(&mut self, r: F) {
        // ---------------- bind coefficients (as in SplitEqPolynomial) ----------------
        if self.E1_len == 1 {
            // E1 is fully bound, so we are binding a variable that affects the outer
            // dimension (A|B). This corresponds to merging pairs of rows in E2.
            let n = self.E2_len / 2;
            for i in 0..n {
                let a = self.E2[2 * i];
                let b = self.E2[2 * i + 1];
                self.E2[i] = a + r * (b - a);
            }
            self.E2_len = n;

            // After merging rows pairwise, the global row index of E2[0] halves as well.
            self.row_start /= 2;
        } else {
            // E1 still has >1 columns, so we bind an inner C-variable (Dao–Thaler column
            // dimension). This halves E1_len and linearly folds pairs of column entries.
            let n = self.E1_len / 2;
            for i in 0..n {
                let a = self.E1[2 * i];
                let b = self.E1[2 * i + 1];
                self.E1[i] = a + r * (b - a);
            }
            self.E1_len = n;

            // Once E1 collapses to a single column, Dao–Thaler reduces to the usual
            // linear-time sumcheck, and E2 simply stores the full Eq evaluations for
            // the remaining outer variables. We fold E1 into E2 in-place.
            if self.E1_len == 1 {
                let scale = self.E1[0];
                self.E2[..self.E2_len]
                    .iter_mut()
                    .for_each(|eval| *eval *= scale);
            }
        }

        // One Eq variable is now bound from this worker’s perspective.
        self.num_vars = self.num_vars.saturating_sub(1);

        // ---------------- remap global index interval ----------------
        //
        // Global Eq indices are treated as a flat array in row-major order.
        // Binding any variable halves the total number of Eq points; each new global
        // index corresponds to floor(old_index / 2). Therefore the worker’s slice
        // [global_start, global_end) maps to:
        //
        //   global_start' = floor(global_start / 2)
        //   global_end'   = floor((global_end - 1) / 2) + 1
        //
        // The (x + 1) >> 1 idiom implements exactly this for unsigned integers.
        self.global_start = self.global_start >> 1;
        self.global_end = (self.global_end + 1) >> 1;

        // Length of this worker’s Eq slice in points also halves, rounded up.
        self.worker_len = (self.worker_len + 1) >> 1;
    }

    pub fn merge(&self) -> DensePolynomial<F> {
        let cols = self.E1_len;
        let mut merged = Vec::new();

        // For each row in this worker's rectangle
        for (row_offset, &e2) in self.E2[..self.E2_len].iter().enumerate() {
            let r = self.row_start + row_offset; // global row index
            let row_first = r * cols;
            let row_last = row_first + cols;

            // Intersection of this row with [global_start, global_end)
            let from = self.global_start.max(row_first) - row_first; // col_from
            let to = self.global_end.min(row_last) - row_first; // col_to

            if from >= to {
                continue; // this row contributes no points for this worker
            }

            // Push E2[r] * E1[j] for j in [from, to)
            if cols == 1 {
                // degenerate (no Dao–Thaler) case
                for _j in from..to {
                    merged.push(e2);
                }
            } else {
                for j in from..to {
                    merged.push(e2 * self.E1[j]);
                }
            }
        }

        DensePolynomial::new_padded(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;

    #[test]
    fn bind() {
        const NUM_VARS: usize = 10;
        let mut rng = test_rng();
        let w: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(NUM_VARS)
            .collect();

        let mut regular_eq = DensePolynomial::new(EqPolynomial::evals(&w));
        let mut split_eq = GruenSplitEqPolynomial::new(&w);
        assert_eq!(regular_eq, split_eq.merge());

        for _ in 0..NUM_VARS {
            let r = Fr::random(&mut rng);
            regular_eq.bound_poly_var_bot(&r);
            split_eq.bind(r);

            let merged = split_eq.merge();
            assert_eq!(regular_eq.Z[..regular_eq.len()], merged.Z[..merged.len()]);
        }
    }

    #[test]
    fn equal_old_and_new_split_eq() {
        const NUM_VARS: usize = 15;
        let mut rng = test_rng();
        let w: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(NUM_VARS)
            .collect();

        let mut old_split_eq = SplitEqPolynomial::new(&w);
        let mut new_split_eq = GruenSplitEqPolynomial::new(&w);

        assert_eq!(old_split_eq.get_num_vars(), new_split_eq.get_num_vars());
        assert_eq!(old_split_eq.len(), new_split_eq.len());
        assert_eq!(old_split_eq.E1, *new_split_eq.to_E1_old());
        assert_eq!(old_split_eq.E2, *new_split_eq.E_out_current());
        assert_eq!(old_split_eq.merge(), new_split_eq.merge());
        // Show that they are the same after binding
        for i in (0..NUM_VARS).rev() {
            let r = Fr::random(&mut rng);
            old_split_eq.bind(r);
            new_split_eq.bind(r);
            assert_eq!(old_split_eq.merge(), new_split_eq.merge());
            if NUM_VARS / 2 < i {
                assert_eq!(old_split_eq.E1_len, new_split_eq.E_in_current_len() * 2);
                assert_eq!(old_split_eq.E2_len, new_split_eq.E_out_current_len());
            } else if i > 0 {
                assert_eq!(old_split_eq.E1_len, new_split_eq.E_in_current_len());
                assert_eq!(old_split_eq.E2_len, new_split_eq.E_out_current_len() * 2);
            }
        }
    }

    #[test]
    fn bench_old_and_new_split_eq() {
        let mut rng = test_rng();
        for num_vars in 5..30 {
            let w: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(num_vars)
                .collect();
            println!("Testing for {} variables", num_vars);

            let start_old_split_eq_time = std::time::Instant::now();
            let _old_split_eq = SplitEqPolynomial::new(&w);
            let end_old_split_eq_time = std::time::Instant::now();
            println!(
                "Time taken for creating old split eq: {:?}",
                end_old_split_eq_time.duration_since(start_old_split_eq_time)
            );

            let start_new_split_eq_time = std::time::Instant::now();
            let _new_split_eq = GruenSplitEqPolynomial::new(&w);
            let end_new_split_eq_time = std::time::Instant::now();
            println!(
                "Time taken for creating new split eq: {:?}",
                end_new_split_eq_time.duration_since(start_new_split_eq_time)
            );
        }
    }

    #[test]
    fn test_new_for_small_value() {
        let mut rng = test_rng();
        const N: usize = 10; // Total variables
        const L0: usize = 3; // SVO rounds

        // Test case 1: Standard setup
        let num_x_out_vars_1 = 2; // Example split for x_out part
        let w1: Vec<Fr> = (0..N).map(|i| Fr::from(i as u64)).collect(); // Use predictable values

        let num_x_in_vars_1 = N - num_x_out_vars_1 - L0;
        let split_eq1 =
            GruenSplitEqPolynomial::new_for_small_value(&w1, num_x_out_vars_1, num_x_in_vars_1, L0);

        // Verify split points and variable slices
        let split_point1_expected1 = num_x_out_vars_1; // Should be 2
        let split_point_x_in_expected1 = num_x_out_vars_1 + num_x_in_vars_1;
        assert_eq!(split_eq1.current_index, split_point1_expected1); // repurposed current_index

        let w_E_in_vars_expected1: Vec<Fr> =
            w1[split_point1_expected1..split_point_x_in_expected1].to_vec(); // w[2..7] = [2,3,4,5,6]
        let mut w_E_out_vars_expected1: Vec<Fr> = Vec::new();
        w_E_out_vars_expected1.extend_from_slice(&w1[0..split_point1_expected1]); // w[0..2] = [0,1]
                                                                                  // Suffix slice is w[split_point_x_in .. N-1] = w[7..9] for N=10, L0=3.
        if split_point_x_in_expected1 < N - 1 {
            // Match logic in main code for L0 > 0
            w_E_out_vars_expected1.extend_from_slice(&w1[split_point_x_in_expected1..N - 1]);
            // w[7..9] = [7,8]
        }
        // Combined = [0, 1, 7, 8]

        // Verify E_in content
        assert_eq!(split_eq1.E_in_vec.len(), 1);
        let expected_E_in1 = EqPolynomial::evals(&w_E_in_vars_expected1);
        assert_eq!(split_eq1.E_in_vec[0], expected_E_in1);

        // Verify E_out content (structure and count)
        assert_eq!(split_eq1.E_out_vec.len(), L0); // Should have L0 = 3 vectors

        // Verify E_out content requires understanding evals_cached internal structure
        // evals_cached(w_E_out) returns [ T(w_E_out[0..k], x), T(w_E_out[0..k-1], x), ..., T(w_E_out[0], x), T([], x) ]
        // where k = w_E_out.len(). Let k=4 here ([0,1,7,8]). Returns 5 vectors.
        // new_for_small_value takes the *last* L0=3 vectors and reverses them.
        // Last 3 vectors from evals_cached([0,1,7,8]) correspond to challenges w=[0,1,7], w=[0,1], w=[0]
        // After reversal: E_out_vec[0] is cache for w=[0], E_out_vec[1] for w=[0,1], E_out_vec[2] for w=[0,1,7]

        let cached_E_out1 = EqPolynomial::evals_cached(&w_E_out_vars_expected1);
        // Expected: cached_E_out1 has len k+1 = 5
        assert_eq!(cached_E_out1.len(), w_E_out_vars_expected1.len() + 1);

        // E_out_vec[0] should be cached_E_out1[4] (evals for w=[0])
        assert_eq!(
            split_eq1.E_out_vec[0],
            cached_E_out1[w_E_out_vars_expected1.len() - 0]
        );
        // E_out_vec[1] should be cached_E_out1[3] (evals for w=[0,1])
        assert_eq!(
            split_eq1.E_out_vec[1],
            cached_E_out1[w_E_out_vars_expected1.len() - 1]
        );
        // E_out_vec[2] should be cached_E_out1[2] (evals for w=[0,1,7])
        assert_eq!(
            split_eq1.E_out_vec[2],
            cached_E_out1[w_E_out_vars_expected1.len() - 2]
        );

        // Test case 2: Edge case L0 = 0
        let num_x_out_vars_2 = N / 2; // Max possible value for num_x_out_vars if num_x_in_vars is also N/2 and L0=0
        let w2: Vec<Fr> = (0..N).map(|_| Fr::random(&mut rng)).collect();
        let num_x_in_vars_2 = N - num_x_out_vars_2 - 0; // L0 is 0
        let split_eq2 =
            GruenSplitEqPolynomial::new_for_small_value(&w2, num_x_out_vars_2, num_x_in_vars_2, 0);
        assert_eq!(split_eq2.E_out_vec.len(), 0);
        assert_eq!(split_eq2.E_in_vec.len(), 1); // E_in should cover w[N/2 .. N/2 + num_x_in_vars_2 -1]
        let split_point1_expected2 = num_x_out_vars_2;
        let split_point_x_in_expected2 = num_x_out_vars_2 + num_x_in_vars_2;
        let w_E_in_vars_expected2: Vec<Fr> =
            w2[split_point1_expected2..split_point_x_in_expected2].to_vec();
        assert!(w_E_in_vars_expected2.len() == num_x_in_vars_2);
        let expected_E_in2 = EqPolynomial::evals(&w_E_in_vars_expected2); // evals of N/2 vars
        assert_eq!(split_eq2.E_in_vec[0], expected_E_in2);

        // Test case 3: Panic case N = 0
        let w3: Vec<Fr> = vec![];
        let l0_3 = 0;
        let num_x_out_vars_3 = 0;
        let n3 = w3.len();
        let num_x_in_vars_3 = n3 - num_x_out_vars_3 - l0_3; // 0 - 0 - 0 = 0
        let result3 = std::panic::catch_unwind(|| {
            GruenSplitEqPolynomial::new_for_small_value(
                &w3,
                num_x_out_vars_3,
                num_x_in_vars_3,
                l0_3,
            );
        });
        assert!(result3.is_err());
    }
}
