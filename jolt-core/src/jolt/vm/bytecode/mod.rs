use std::collections::BTreeMap;

use crate::dag::stage::{StagedSumcheck, SumcheckStages};
use crate::dag::state_manager::StateManager;
use crate::jolt::vm::bytecode::booleanity::BooleanitySumcheck;
use crate::jolt::vm::bytecode::hamming_weight::HammingWeightSumcheck;
use crate::jolt::vm::bytecode::raf::RafBytecode;
use crate::jolt::vm::bytecode::read_checking::{ReadCheckingSumcheck, ReadCheckingValType};
use crate::poly::opening_proof::OpeningsKeys;
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::utils::math::Math;
use crate::{
    field::JoltField,
    jolt::witness::CommittedPolynomials,
    poly::{commitment::commitment_scheme::CommitmentScheme, eq_poly::EqPolynomial},
    utils::{thread::unsafe_allocate_zero_vec, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS};
use rayon::prelude::*;
use tracer::instruction::{RV32IMCycle, RV32IMInstruction};

pub mod booleanity;
pub mod hamming_weight;
pub mod raf;
pub mod read_checking;

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodePreprocessing {
    pub code_size: usize,
    pub bytecode: Vec<RV32IMInstruction>,
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    /// Key: (ELF address, virtual sequence index or 0)
    pub virtual_address_map: BTreeMap<(usize, usize), usize>,
    pub d: usize,
}

impl BytecodePreprocessing {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<RV32IMInstruction>) -> Self {
        let mut virtual_address_map = BTreeMap::new();
        let mut virtual_address = 1; // Account for no-op instruction prepended to bytecode
        for instruction in bytecode.iter() {
            if instruction.normalize().address == 0 {
                // ignore unimplemented instructions
                continue;
            }
            let instr = instruction.normalize();
            debug_assert!(instr.address >= RAM_START_ADDRESS as usize);
            debug_assert!(instr.address.is_multiple_of(BYTES_PER_INSTRUCTION));
            assert_eq!(
                virtual_address_map.insert(
                    (instr.address, instr.virtual_sequence_remaining.unwrap_or(0)),
                    virtual_address
                ),
                None
            );
            virtual_address += 1;
        }

        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, RV32IMInstruction::NoOp);
        assert_eq!(virtual_address_map.insert((0, 0), 0), None);

        // Bytecode: Pad to nearest power of 2
        let code_size = bytecode.len().next_power_of_two();
        bytecode.resize(code_size, RV32IMInstruction::NoOp);

        // TODO: change this to have different calculations
        let d = 2;

        Self {
            code_size,
            bytecode,
            virtual_address_map,
            d,
        }
    }

    pub fn get_pc(&self, cycle: &RV32IMCycle) -> usize {
        if matches!(cycle, tracer::instruction::RV32IMCycle::NoOp) {
            return 0;
        }
        let instr = cycle.instruction().normalize();
        *self
            .virtual_address_map
            .get(&(instr.address, instr.virtual_sequence_remaining.unwrap_or(0)))
            .unwrap()
    }
}

#[derive(Default)]
pub struct BytecodeDag {}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, T: Transcript> SumcheckStages<F, T, PCS>
    for BytecodeDag
{
    fn stage4_prover_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let (preprocessing, trace, _, _) = sm.get_prover_data();
        let bytecode_preprocessing = &preprocessing.shared.bytecode;

        let r_cycle_1: Vec<F> = sm
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::UnexpandedPC))
            .unwrap()
            .r;
        let r_cycle_2 = sm
            .get_opening_point(OpeningsKeys::RegistersReadWriteInc)
            .unwrap()
            .r;
        let r_cycle_3 = sm
            .get_opening_point(OpeningsKeys::RegistersValEvaluationInc)
            .unwrap()
            .r;
        let E_1: Vec<F> = EqPolynomial::evals(&r_cycle_1);
        let E_2: Vec<F> = EqPolynomial::evals(&r_cycle_2);
        let E_3: Vec<F> = EqPolynomial::evals(&r_cycle_3);

        let (F_1, F_2, F_3) = compute_ra_evals(bytecode_preprocessing, trace, &E_1, &E_2, &E_3);

        let d = bytecode_preprocessing.d;
        let unbound_ra_polys = (0..d)
            .map(|i| CommittedPolynomials::BytecodeRa(i).generate_witness(preprocessing, trace))
            .collect::<Vec<_>>();

        // let read_checking_1 = ReadCheckingSumcheck::new_prover(
        //     sm,
        //     F_1.clone(),
        //     unbound_ra_poly.clone(),
        //     ReadCheckingValType::Stage1,
        // );
        // let read_checking_2 = ReadCheckingSumcheck::new_prover(
        //     sm,
        //     F_2,
        //     unbound_ra_poly.clone(),
        //     ReadCheckingValType::Stage2,
        // );
        // let read_checking_3 = ReadCheckingSumcheck::new_prover(
        //     sm,
        //     F_3.clone(),
        //     unbound_ra_poly.clone(),
        //     ReadCheckingValType::Stage3,
        // );
        // let raf = RafBytecode::new_prover(
        //     sm,
        //     MultilinearPolynomial::from(F_1.clone()),
        //     MultilinearPolynomial::from(F_3),
        // );
        let booleanity = BooleanitySumcheck::new_prover(sm, E_1, F_1.clone());
        let hamming_weight = HammingWeightSumcheck::new_prover(sm, F_1, unbound_ra_polys);

        vec![
            // Box::new(read_checking_1),
            // Box::new(read_checking_2),
            // Box::new(read_checking_3),
            // Box::new(raf),
            Box::new(booleanity),
            Box::new(hamming_weight),
        ]
    }

    fn stage4_verifier_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        // let read_checking_1 = ReadCheckingSumcheck::new_verifier(sm, ReadCheckingValType::Stage1);
        // let read_checking_2 = ReadCheckingSumcheck::new_verifier(sm, ReadCheckingValType::Stage2);
        // let read_checking_3 = ReadCheckingSumcheck::new_verifier(sm, ReadCheckingValType::Stage3);
        // let raf = RafBytecode::new_verifier(sm);
        let booleanity = BooleanitySumcheck::new_verifier(sm);
        let hamming_weight = HammingWeightSumcheck::new_verifier(sm);

        vec![
            // Box::new(read_checking_1),
            // Box::new(read_checking_2),
            // Box::new(read_checking_3),
            // Box::new(raf),
            Box::new(booleanity),
            Box::new(hamming_weight),
        ]
    }
}

#[inline(always)]
#[tracing::instrument(skip_all, name = "Bytecode::compute_ra_evals")]
fn compute_ra_evals<F: JoltField>(
    preprocessing: &BytecodePreprocessing,
    trace: &[RV32IMCycle],
    eq_r_1: &[F],
    eq_r_2: &[F],
    eq_r_3: &[F],
) -> (Vec<Vec<F>>, Vec<Vec<F>>, Vec<Vec<F>>) {
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);
    let log_K = preprocessing.code_size.log_2();
    let d = preprocessing.d;
    let log_K_chunk = log_K.div_ceil(d);
    let K_chunk = log_K_chunk.pow2();

    trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result_1: Vec<Vec<F>> =
                (0..d).map(|_| unsafe_allocate_zero_vec(K_chunk)).collect();
            let mut result_2: Vec<Vec<F>> =
                (0..d).map(|_| unsafe_allocate_zero_vec(K_chunk)).collect();
            let mut result_3: Vec<Vec<F>> =
                (0..d).map(|_| unsafe_allocate_zero_vec(K_chunk)).collect();
            let mut j = chunk_index * chunk_size;
            for cycle in trace_chunk {
                let mut pc = preprocessing.get_pc(cycle);
                for i in (0..d).rev() {
                    let k = pc % K_chunk;
                    result_1[i][k] += eq_r_1[j];
                    result_2[i][k] += eq_r_2[j];
                    result_3[i][k] += eq_r_3[j];
                    pc >>= log_K_chunk;
                }
                j += 1;
            }
            (result_1, result_2, result_3)
        })
        .reduce(
            || {
                (
                    (0..d)
                        .map(|_| unsafe_allocate_zero_vec(K_chunk))
                        .collect::<Vec<_>>(),
                    (0..d)
                        .map(|_| unsafe_allocate_zero_vec(K_chunk))
                        .collect::<Vec<_>>(),
                    (0..d)
                        .map(|_| unsafe_allocate_zero_vec(K_chunk))
                        .collect::<Vec<_>>(),
                )
            },
            |(mut running_1, mut running_2, mut running_3), (new_1, new_2, new_3)| {
                running_1
                    .par_iter_mut()
                    .zip(new_1.into_par_iter())
                    .for_each(|(x, y)| {
                        x.par_iter_mut()
                            .zip(y.into_par_iter())
                            .for_each(|(x, y)| *x += y)
                    });
                running_2
                    .par_iter_mut()
                    .zip(new_2.into_par_iter())
                    .for_each(|(x, y)| {
                        x.par_iter_mut()
                            .zip(y.into_par_iter())
                            .for_each(|(x, y)| *x += y)
                    });
                running_3
                    .par_iter_mut()
                    .zip(new_3.into_par_iter())
                    .for_each(|(x, y)| {
                        x.par_iter_mut()
                            .zip(y.into_par_iter())
                            .for_each(|(x, y)| *x += y)
                    });
                (running_1, running_2, running_3)
            },
        )
}
