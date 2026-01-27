use crate::constants::{
    DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
    DEFAULT_MEMORY_SIZE, DEFAULT_STACK_SIZE, MEMORY_OPS_PER_INSTRUCTION, RAM_START_ADDRESS,
    REGISTER_COUNT,
};
#[cfg(not(feature = "std"))]
use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use core::str::FromStr;
use serde::{Deserialize, Serialize};
use strum::EnumCount;
use strum_macros::{AsRefStr, EnumCount as EnumCountMacro, EnumIter, FromRepr};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RVTraceRow {
    pub instruction: ELFInstruction,
    pub register_state: RegisterState,
    pub memory_state: Option<MemoryState>,
    pub advice_value: Option<u64>,
    pub precompile_input: Option<[u32; 16]>,
    pub precompile_output_address: Option<u64>,
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryOp {
    Read(u64),       // (address)
    Write(u64, u64), // (address, new_value)
}

impl CanonicalSerialize for MemoryOp {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            MemoryOp::Read(address) => {
                (0_u8).serialize_with_mode(&mut writer, compress)?;
                address.serialize_with_mode(&mut writer, compress)?;
            }
            MemoryOp::Write(address, value) => {
                (1_u8).serialize_with_mode(&mut writer, compress)?;
                address.serialize_with_mode(&mut writer, compress)?;
                value.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            MemoryOp::Read(address) => {
                (0_u8).serialized_size(compress) + address.serialized_size(compress)
            }
            MemoryOp::Write(address, value) => {
                (1_u8).serialized_size(compress)
                    + address.serialized_size(compress)
                    + value.serialized_size(compress)
            }
        }
    }
}

impl CanonicalDeserialize for MemoryOp {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        // TODO(protoben) Can we use strum for this?
        let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let res = match discriminant {
            0 => MemoryOp::Read(u64::deserialize_with_mode(&mut reader, compress, validate)?),
            1 => MemoryOp::Write(
                u64::deserialize_with_mode(&mut reader, compress, validate)?,
                u64::deserialize_with_mode(&mut reader, compress, validate)?,
            ),
            _ => Err(SerializationError::InvalidData)?,
        };
        Ok(res)
    }
}

impl Valid for MemoryOp {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            MemoryOp::Read(inner) => inner.check(),
            MemoryOp::Write(address, new_value) => {
                address.check()?;
                new_value.check()?;
                Ok(())
            }
        }
    }
}

impl MemoryOp {
    pub fn noop_read() -> Self {
        Self::Read(0)
    }

    pub fn noop_write() -> Self {
        Self::Write(0, 0)
    }
}

fn sum_u64_i32(a: u64, b: i32) -> u64 {
    if b.is_negative() {
        let abs_b = b.unsigned_abs() as u64;
        if a < abs_b {
            panic!("overflow")
        }
        a - abs_b
    } else {
        let b_u64: u64 = b.try_into().expect("failed u64 conversion");
        a + b_u64
    }
}

impl From<&RVTraceRow> for [MemoryOp; MEMORY_OPS_PER_INSTRUCTION] {
    fn from(val: &RVTraceRow) -> Self {
        let rs1_read = || MemoryOp::Read(val.instruction.rs1.unwrap());
        let rs2_read = || MemoryOp::Read(val.instruction.rs2.unwrap());
        let rd_write = || {
            MemoryOp::Write(
                val.instruction.rd.unwrap(),
                val.register_state.rd_post_val.unwrap(),
            )
        };

        let ram_write_value = || match val.memory_state {
            Some(MemoryState::Read {
                address: _,
                value: _,
            }) => panic!("Unexpected MemoryState::Read"),
            Some(MemoryState::Write {
                address: _,
                pre_value: _,
                post_value,
            }) => post_value,
            None => panic!("Memory state not found"),
        };

        let rs1_offset = || -> u64 {
            let rs1_val = val.register_state.rs1_val.unwrap();
            let imm = val.instruction.imm.unwrap();
            sum_u64_i32(rs1_val, imm as i32)
        };

        // Canonical ordering for memory instructions
        // 0: rs1
        // 1: rs2
        // 2: rd
        // 3: byte_0
        // 4: byte_1
        // 5: byte_2
        // 6: byte_3
        // If any are empty a no_op is inserted.

        match val.instruction.opcode {
            RV32IM::ADD
            | RV32IM::SUB
            | RV32IM::XOR
            | RV32IM::OR
            | RV32IM::AND
            | RV32IM::SLL
            | RV32IM::SRL
            | RV32IM::SRA
            | RV32IM::SLT
            | RV32IM::SLTU
            | RV32IM::MUL
            | RV32IM::MULH
            | RV32IM::MULHU
            | RV32IM::MULHSU
            | RV32IM::MULU
            | RV32IM::DIV
            | RV32IM::DIVU
            | RV32IM::REM
            | RV32IM::REMU => [rs1_read(), rs2_read(), rd_write(), MemoryOp::noop_read()],

            RV32IM::LUI | RV32IM::AUIPC | RV32IM::VIRTUAL_ADVICE => [
                MemoryOp::noop_read(),
                MemoryOp::noop_read(),
                rd_write(),
                MemoryOp::noop_read(),
            ],

            RV32IM::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT => [
                rs1_read(),
                MemoryOp::noop_read(),
                MemoryOp::noop_write(),
                MemoryOp::noop_read(),
            ],

            RV32IM::ADDI
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI
            | RV32IM::ANDI
            | RV32IM::ORI
            | RV32IM::XORI
            | RV32IM::SLTI
            | RV32IM::SLTIU
            | RV32IM::JALR
            | RV32IM::VIRTUAL_MOVE
            | RV32IM::VIRTUAL_MOVSIGN => [
                rs1_read(),
                MemoryOp::noop_read(),
                rd_write(),
                MemoryOp::noop_read(),
            ],

            RV32IM::LW => [
                rs1_read(),
                MemoryOp::noop_read(),
                rd_write(),
                MemoryOp::Read(rs1_offset()),
            ],
            RV32IM::FENCE => [
                MemoryOp::noop_read(),
                MemoryOp::noop_read(),
                MemoryOp::noop_write(),
                MemoryOp::noop_read(),
            ],

            RV32IM::SB | RV32IM::SH | RV32IM::SW => [
                rs1_read(),
                rs2_read(),
                MemoryOp::noop_write(),
                MemoryOp::Write(rs1_offset(), ram_write_value()),
            ],

            // RV32IM::LB | RV32IM::LH | RV32IM::LBU | RV32IM::LHU => [
            RV32IM::JAL => [
                MemoryOp::noop_read(),
                MemoryOp::noop_read(),
                rd_write(),
                MemoryOp::noop_read(),
            ],

            RV32IM::BEQ
            | RV32IM::BNE
            | RV32IM::BLT
            | RV32IM::BGE
            | RV32IM::BLTU
            | RV32IM::BGEU
            | RV32IM::VIRTUAL_ASSERT_EQ
            | RV32IM::VIRTUAL_ASSERT_LTE
            | RV32IM::VIRTUAL_ASSERT_VALID_DIV0
            | RV32IM::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER
            | RV32IM::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER => [
                rs1_read(),
                rs2_read(),
                MemoryOp::noop_write(),
                MemoryOp::noop_read(),
            ],

            RV32IM::ECALL => [
                MemoryOp::noop_read(),
                MemoryOp::noop_read(),
                MemoryOp::noop_write(),
                MemoryOp::Write(rs1_offset(), ram_write_value()),
            ],

            _ => unreachable!("{val:?}"),
        }
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ELFInstruction {
    pub address: u64,
    pub opcode: RV32IM,
    pub rs1: Option<u64>,
    pub rs2: Option<u64>,
    pub rd: Option<u64>,
    pub imm: Option<i64>,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl core::fmt::Debug for ELFInstruction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ELFInstruction {{ address: {:X}, opcode: {:?}, rs1: {:?}, rs2: {:?}, rd: {:?}, imm: {:?}, virtual_sequence_remaining: {:?} }}",
            self.address, self.opcode, self.rs1, self.rs2, self.rd, self.imm, self.virtual_sequence_remaining
        )
    }
}

/// Boolean flags used in Jolt's R1CS constraints (`opflags` in the Jolt paper).
/// Note that the flags below deviate slightly from those described in Appendix A.1
/// of the Jolt paper.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Hash, Ord, EnumCountMacro, EnumIter, Default,
)]
pub enum CircuitFlags {
    #[default] // Need a default so that we can derive EnumIter on `JoltR1CSInputs`
    /// 1 if the first lookup operand is the program counter; 0 otherwise (first lookup operand is RS1 value).
    LeftOperandIsPC,
    /// 1 if the second lookup operand is `imm`; 0 otherwise (second lookup operand is RS2 value).
    RightOperandIsImm,
    /// 1 if the instruction is a load (i.e. `LW`)
    Load,
    /// 1 if the instruction is a store (i.e. `SW`)
    Store,
    /// 1 if the instruction is a jump (i.e. `JAL`, `JALR`)
    Jump,
    /// 1 if the instruction is a branch (i.e. `BEQ`, `BNE`, etc.)
    Branch,
    Lui,
    /// 1 if the lookup output is to be stored in `rd` at the end of the step.
    WriteLookupOutputToRD,
    /// Indicates whether the instruction performs a concat-type lookup.
    ConcatLookupQueryChunks,
    /// 1 if the instruction is "virtual", as defined in Section 6.1 of the Jolt paper.
    Virtual,
    /// 1 if the instruction is an assert, as defined in Section 6.1.1 of the Jolt paper.
    Assert,
    /// Used in virtual sequences; the program counter should be the same for the full sequence.
    DoNotUpdatePC,
}
pub const NUM_CIRCUIT_FLAGS: usize = CircuitFlags::COUNT;

impl ELFInstruction {
    #[rustfmt::skip]
    pub fn to_circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];

        flags[CircuitFlags::LeftOperandIsPC as usize] = matches!(
            self.opcode,
            RV32IM::JAL | RV32IM::AUIPC,
        );

        flags[CircuitFlags::RightOperandIsImm as usize] = matches!(
            self.opcode,
            RV32IM::ADDI
            | RV32IM::XORI
            | RV32IM::ORI
            | RV32IM::ANDI
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI
            | RV32IM::SLTI
            | RV32IM::SLTIU
            | RV32IM::AUIPC
            | RV32IM::JAL
            | RV32IM::JALR
            | RV32IM::SW
            | RV32IM::LW
            | RV32IM::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT,
        );

        flags[CircuitFlags::Load as usize] = matches!(
            self.opcode,
            RV32IM::LW,
        );

        flags[CircuitFlags::Store as usize] = matches!(
            self.opcode,
            RV32IM::SW,
        );

        flags[CircuitFlags::Jump as usize] = matches!(
            self.opcode,
            RV32IM::JAL | RV32IM::JALR,
        );

        flags[CircuitFlags::Branch as usize] = matches!(
            self.opcode,
            RV32IM::BEQ | RV32IM::BNE | RV32IM::BLT | RV32IM::BGE | RV32IM::BLTU | RV32IM::BGEU,
        );

        flags[CircuitFlags::Lui as usize] = matches!(
            self.opcode,
            RV32IM::LUI,
        );

        // Stores, branches, jumps, and asserts do not store the lookup output to rd (they may update rd in other ways)
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = !matches!(
            self.opcode,
            RV32IM::SW
            | RV32IM::LW
            | RV32IM::BEQ
            | RV32IM::BNE
            | RV32IM::BLT
            | RV32IM::BGE
            | RV32IM::BLTU
            | RV32IM::BGEU
            | RV32IM::JAL
            | RV32IM::JALR
            | RV32IM::LUI
            | RV32IM::VIRTUAL_ASSERT_EQ
            | RV32IM::VIRTUAL_ASSERT_LTE
            | RV32IM::VIRTUAL_ASSERT_VALID_DIV0
            | RV32IM::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER
            | RV32IM::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER
            | RV32IM::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT
        );

        flags[CircuitFlags::ConcatLookupQueryChunks as usize] = matches!(
            self.opcode,
            RV32IM::XOR
            | RV32IM::XORI
            | RV32IM::OR
            | RV32IM::ORI
            | RV32IM::AND
            | RV32IM::ANDI
            | RV32IM::SLL
            | RV32IM::SRL
            | RV32IM::SRA
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI
            | RV32IM::SLT
            | RV32IM::SLTU
            | RV32IM::SLTI
            | RV32IM::SLTIU
            | RV32IM::BEQ
            | RV32IM::BNE
            | RV32IM::BLT
            | RV32IM::BGE
            | RV32IM::BLTU
            | RV32IM::BGEU
            | RV32IM::VIRTUAL_ASSERT_EQ
            | RV32IM::VIRTUAL_ASSERT_LTE
            | RV32IM::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER
            | RV32IM::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER
            | RV32IM::VIRTUAL_ASSERT_VALID_DIV0,
        );

        flags[CircuitFlags::Virtual as usize] = self.virtual_sequence_remaining.is_some();

        flags[CircuitFlags::Assert as usize] = matches!(self.opcode,
            RV32IM::VIRTUAL_ASSERT_EQ                        |
            RV32IM::VIRTUAL_ASSERT_LTE                       |
            RV32IM::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT        |
            RV32IM::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER    |
            RV32IM::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER  |
            RV32IM::VIRTUAL_ASSERT_VALID_DIV0
        );

        // All instructions in virtual sequence are mapped from the same
        // ELF address. Thus if an instruction is virtual (and not the last one
        // in its sequence), then we should *not* update the PC.
        flags[CircuitFlags::DoNotUpdatePC as usize] = match self.virtual_sequence_remaining {
            Some(i) => i != 0,
            None => false
        };

        flags
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RegisterState {
    pub rs1_val: Option<u64>,
    pub rs2_val: Option<u64>,
    pub rd_post_val: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MemoryState {
    Read {
        address: u64,
        value: u64,
    },
    Write {
        address: u64,
        pre_value: u64,
        post_value: u64,
    },
}

impl RVTraceRow {
    pub fn imm_u64(&self) -> u64 {
        self.instruction.imm.unwrap() as u64
    }

    pub fn imm_u32(&self) -> u32 {
        self.instruction.imm.unwrap() as u64 as u32
    }
}

// Reference: https://www.cs.sfu.ca/~ashriram/Courses/CS295/assets/notebooks/RISCV/RISCV_CARD.pdf
#[derive(
    Debug,
    PartialEq,
    Eq,
    Clone,
    Copy,
    FromRepr,
    Serialize,
    Deserialize,
    Hash,
    PartialOrd,
    Ord,
    AsRefStr,
)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum RV32IM {
    ADD,
    SUB,
    XOR,
    OR,
    AND,
    SLL,
    SRL,
    SRA,
    SLT,
    SLTU,
    ADDI,
    XORI,
    ORI,
    ANDI,
    SLLI,
    SRLI,
    SRAI,
    SLTI,
    SLTIU,
    LB,
    LH,
    LW,
    LBU,
    LHU,
    SB,
    SH,
    SW,
    BEQ,
    BNE,
    BLT,
    BGE,
    BLTU,
    BGEU,
    JAL,
    JALR,
    LUI,
    AUIPC,
    ECALL,
    EBREAK,
    MUL,
    MULH,
    MULHU,
    MULHSU,
    MULU,
    DIV,
    DIVU,
    REM,
    REMU,
    FENCE,
    UNIMPL,
    // Virtual instructions
    VIRTUAL_MOVSIGN,
    VIRTUAL_MOVE,
    VIRTUAL_ADVICE,
    VIRTUAL_ASSERT_LTE,
    VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER,
    VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER,
    VIRTUAL_ASSERT_EQ,
    VIRTUAL_ASSERT_VALID_DIV0,
    VIRTUAL_ASSERT_HALFWORD_ALIGNMENT,
    VIRTUAL_POW2,
    VIRTUAL_POW2I,
    VIRTUAL_SRA_PAD,
    VIRTUAL_SRA_PADI,
}

impl FromStr for RV32IM {
    type Err = String;

    fn from_str(s: &str) -> Result<RV32IM, String> {
        match s {
            "ADD" => Ok(Self::ADD),
            "SUB" => Ok(Self::SUB),
            "XOR" => Ok(Self::XOR),
            "OR" => Ok(Self::OR),
            "AND" => Ok(Self::AND),
            "SLL" => Ok(Self::SLL),
            "SRL" => Ok(Self::SRL),
            "SRA" => Ok(Self::SRA),
            "SLT" => Ok(Self::SLT),
            "SLTU" => Ok(Self::SLTU),
            "ADDI" => Ok(Self::ADDI),
            "XORI" => Ok(Self::XORI),
            "ORI" => Ok(Self::ORI),
            "ANDI" => Ok(Self::ANDI),
            "SLLI" => Ok(Self::SLLI),
            "SRLI" => Ok(Self::SRLI),
            "SRAI" => Ok(Self::SRAI),
            "SLTI" => Ok(Self::SLTI),
            "SLTIU" => Ok(Self::SLTIU),
            "LB" => Ok(Self::LB),
            "LH" => Ok(Self::LH),
            "LW" => Ok(Self::LW),
            "LBU" => Ok(Self::LBU),
            "LHU" => Ok(Self::LHU),
            "SB" => Ok(Self::SB),
            "SH" => Ok(Self::SH),
            "SW" => Ok(Self::SW),
            "BEQ" => Ok(Self::BEQ),
            "BNE" => Ok(Self::BNE),
            "BLT" => Ok(Self::BLT),
            "BGE" => Ok(Self::BGE),
            "BLTU" => Ok(Self::BLTU),
            "BGEU" => Ok(Self::BGEU),
            "JAL" => Ok(Self::JAL),
            "JALR" => Ok(Self::JALR),
            "LUI" => Ok(Self::LUI),
            "AUIPC" => Ok(Self::AUIPC),
            "ECALL" => Ok(Self::ECALL),
            "EBREAK" => Ok(Self::EBREAK),
            "MUL" => Ok(Self::MUL),
            "MULH" => Ok(Self::MULH),
            "MULHU" => Ok(Self::MULHU),
            "MULHSU" => Ok(Self::MULHSU),
            "MULU" => Ok(Self::MULU),
            "DIV" => Ok(Self::DIV),
            "DIVU" => Ok(Self::DIVU),
            "REM" => Ok(Self::REM),
            "REMU" => Ok(Self::REMU),
            "FENCE" => Ok(Self::FENCE),
            "UNIMPL" => Ok(Self::UNIMPL),
            _ => Err("Could not match instruction to RV32IM set.".to_string()),
        }
    }
}

#[allow(clippy::too_long_first_doc_paragraph)]
/// Represented as a "peripheral device" in the RISC-V emulator, this captures
/// all reads from the reserved memory address space for program inputs and all writes
/// to the reserved memory address space for program outputs.
/// The inputs and outputs are part of the public inputs to the proof.
#[derive(
    Debug, Clone, PartialEq, Serialize, Deserialize, CanonicalSerialize, CanonicalDeserialize,
)]
pub struct JoltDevice {
    pub inputs: Vec<u8>,
    // pub trusted_advice: Vec<u8>,
    pub untrusted_advice: Vec<u8>,
    pub outputs: Vec<u8>,
    pub panic: bool,
    pub memory_layout: MemoryLayout,
}

impl JoltDevice {
    pub fn new(memory_config: &MemoryConfig) -> Self {
        Self {
            inputs: Vec::new(),
            // trusted_advice: Vec::new(),
            untrusted_advice: Vec::new(),
            outputs: Vec::new(),
            panic: false,
            memory_layout: MemoryLayout::new(memory_config),
        }
    }

    pub fn load(&self, address: u64) -> u8 {
        if self.is_panic(address) {
            self.panic as u8
        } else if self.is_termination(address) {
            0 // Termination bit should never be loaded after it is set
        } else if self.is_input(address) {
            let internal_address = self.convert_read_address(address);
            if self.inputs.len() <= internal_address {
                0
            } else {
                self.inputs[internal_address]
            }
        // } else if self.is_trusted_advice(address) {
        //     let internal_address = self.convert_trusted_advice_read_address(address);
        //     if self.trusted_advice.len() <= internal_address {
        //         0
        //     } else {
        //         self.trusted_advice[internal_address]
        //     }
        } else if self.is_untrusted_advice(address) {
            let internal_address = self.convert_untrusted_advice_read_address(address);
            if self.untrusted_advice.len() <= internal_address {
                0
            } else {
                self.untrusted_advice[internal_address]
            }
        } else if self.is_output(address) {
            let internal_address = self.convert_write_address(address);
            if self.outputs.len() <= internal_address {
                0
            } else {
                self.outputs[internal_address]
            }
        } else {
            assert!(address <= RAM_START_ADDRESS - 8);
            0 // zero-padding
        }
    }

    pub fn store(&mut self, address: u64, value: u8) {
        if address == self.memory_layout.panic {
            #[cfg(feature = "std")]
            println!("GUEST PANIC");
            self.panic = true;
            return;
        }

        if address == self.memory_layout.termination {
            return;
        }

        let internal_address = self.convert_write_address(address);
        if self.outputs.len() <= internal_address {
            self.outputs.resize(internal_address + 1, 0);
        }

        self.outputs[internal_address] = value;
    }

    pub fn size(&self) -> usize {
        self.inputs.len() + self.outputs.len()
    }

    pub fn is_input(&self, address: u64) -> bool {
        address >= self.memory_layout.input_start && address < self.memory_layout.input_end
    }

    // pub fn is_trusted_advice(&self, address: u64) -> bool {
    //     address >= self.memory_layout.trusted_advice_start
    //         && address < self.memory_layout.trusted_advice_end
    // }

    pub fn is_untrusted_advice(&self, address: u64) -> bool {
        address >= self.memory_layout.untrusted_advice_start
            && address < self.memory_layout.untrusted_advice_end
    }

    pub fn is_output(&self, address: u64) -> bool {
        address >= self.memory_layout.output_start && address < self.memory_layout.termination
    }

    pub fn is_panic(&self, address: u64) -> bool {
        address == self.memory_layout.panic
    }

    pub fn is_termination(&self, address: u64) -> bool {
        address == self.memory_layout.termination
    }

    fn convert_read_address(&self, address: u64) -> usize {
        (address - self.memory_layout.input_start) as usize
    }

    fn convert_write_address(&self, address: u64) -> usize {
        (address - self.memory_layout.output_start) as usize
    }

    // fn convert_trusted_advice_read_address(&self, address: u64) -> usize {
    //     (address - self.memory_layout.trusted_advice_start) as usize
    // }

    fn convert_untrusted_advice_read_address(&self, address: u64) -> usize {
        (address - self.memory_layout.untrusted_advice_start) as usize
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MemoryConfig {
    pub max_input_size: u64,
    // pub max_trusted_advice_size: u64,
    pub max_untrusted_advice_size: u64,
    pub max_output_size: u64,
    pub stack_size: u64,
    pub memory_size: u64,
    // pub program_size: Option<u64>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_input_size: DEFAULT_MAX_INPUT_SIZE,
            // max_trusted_advice_size: DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
            max_untrusted_advice_size: DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
            max_output_size: DEFAULT_MAX_OUTPUT_SIZE,
            stack_size: DEFAULT_STACK_SIZE,
            memory_size: DEFAULT_MEMORY_SIZE,
            // program_size: None,
        }
    }
}

#[derive(
    Clone,
    Copy,
    PartialEq,
    Serialize,
    Deserialize,
    CanonicalSerialize,
    CanonicalDeserialize,
    Default,
)]
pub struct MemoryLayout {
    /// The total size of the elf's sections, including the .text, .data, .rodata, and .bss sections.
    // pub program_size: u64,
    // pub max_trusted_advice_size: u64,
    // pub trusted_advice_start: u64,
    // pub trusted_advice_end: u64,
    pub max_untrusted_advice_size: u64,
    pub untrusted_advice_start: u64,
    pub untrusted_advice_end: u64,
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub input_start: u64,
    pub input_end: u64,
    pub output_start: u64,
    pub output_end: u64,
    pub stack_size: u64,
    /// Stack starts from (RAM_START_ADDRESS + `program_size` + `stack_size`) and grows in descending addresses by `stack_size` bytes.
    pub stack_end: u64,
    pub memory_size: u64,
    /// Heap starts just after the start of the stack and is `memory_size` bytes.
    pub memory_end: u64,
    pub panic: u64,
    pub termination: u64,
    // /// End of the memory region containing inputs, outputs, the panic bit,
    // /// and the termination bit
    // pub io_end: u64,
}

impl core::fmt::Debug for MemoryLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MemoryLayout")
            // .field("program_size", &self.program_size)
            .field("max_input_size", &self.max_input_size)
            // .field("max_trusted_advice_size", &self.max_trusted_advice_size)
            .field("max_untrusted_advice_size", &self.max_untrusted_advice_size)
            .field("max_output_size", &self.max_output_size)
            // .field(
            //     "trusted_advice_start",
            //     &format_args!("{:#X}", self.trusted_advice_start),
            // )
            // .field(
            //     "trusted_advice_end",
            //     &format_args!("{:#X}", self.trusted_advice_end),
            // )
            .field(
                "untrusted_advice_start",
                &format_args!("{:#X}", self.untrusted_advice_start),
            )
            .field(
                "untrusted_advice_end",
                &format_args!("{:#X}", self.untrusted_advice_end),
            )
            .field("input_start", &format_args!("{:#X}", self.input_start))
            .field("input_end", &format_args!("{:#X}", self.input_end))
            .field("output_start", &format_args!("{:#X}", self.output_start))
            .field("output_end", &format_args!("{:#X}", self.output_end))
            .field("stack_size", &format_args!("{:#X}", self.stack_size))
            .field("stack_end", &format_args!("{:#X}", self.stack_end))
            .field("memory_size", &format_args!("{:#X}", self.memory_size))
            .field("memory_end", &format_args!("{:#X}", self.memory_end))
            .field("panic", &format_args!("{:#X}", self.panic))
            .field("termination", &format_args!("{:#X}", self.termination))
            .finish()
    }
}

impl MemoryLayout {
    pub fn new(config: &MemoryConfig) -> Self {
        // assert!(
        //     config.program_size.is_some(),
        //     "MemoryLayout requires bytecode size to be set"
        // );
        // helper to align ‘val’ *up* to a multiple of ‘align’, panicking on overflow
        #[inline]
        fn align_up(val: u64, align: u64) -> u64 {
            if align == 0 {
                val
            } else {
                match val % align {
                    0 => val,
                    rem => {
                        // panics if val + (align - rem) overflows
                        val.checked_add(align - rem).expect("alignment overflow")
                    }
                }
            }
        } // Must be 8-byte aligned

        // let max_trusted_advice_size = align_up(config.max_trusted_advice_size, 8);
        let max_untrusted_advice_size = align_up(config.max_untrusted_advice_size, 8);
        let max_input_size = align_up(config.max_input_size, 8);
        let max_output_size = align_up(config.max_output_size, 8);
        let stack_size = align_up(config.stack_size, 8);
        let memory_size = align_up(config.memory_size, 8);

        // Adds 16 to account for panic bit and termination bit
        // (they each occupy one full 8-byte word)
        let io_region_bytes = max_input_size
            .checked_add(max_untrusted_advice_size)
            // .and_then(|s| s.checked_add(max_untrusted_advice_size))
            .and_then(|s| s.checked_add(max_output_size))
            .and_then(|s| s.checked_add(16))
            .expect("I/O region size overflow");

        // Padded so that the witness index corresponding to `input_start`
        // has the form 0b11...100...0
        let io_region_words = (io_region_bytes / 8).next_power_of_two();
        // let io_region_words = (io_region_bytes / 8 + 1).next_power_of_two() - 1;

        let io_bytes = io_region_words
            .checked_mul(8)
            .expect("I/O region byte count overflow");

        // let trusted_advice_start = RAM_START_ADDRESS
        //     .checked_sub(io_bytes)
        //     .expect("I/O region exceeds RAM_START_ADDRESS");
        // let trusted_advice_end = trusted_advice_start
        //     .checked_add(max_trusted_advice_size)
        //     .expect("trusted_advice_end overflow");

        let untrusted_advice_start = RAM_START_ADDRESS
            .checked_sub(io_bytes)
            .expect("I/O region exceeds RAM_START_ADDRESS");
        let untrusted_advice_end = untrusted_advice_start
            .checked_add(max_untrusted_advice_size)
            .expect("untrusted_advice_end overflow");

        let input_start = untrusted_advice_end;
        let input_end = input_start
            .checked_add(max_input_size)
            .expect("input_end overflow");
        let output_start = input_end;
        let output_end = output_start
            .checked_add(max_output_size)
            .expect("output_end overflow");
        let panic = output_end;
        let termination = panic.checked_add(4).expect("termination overflow");

        // stack grows *down* from input_start
        let stack_end = input_start
            .checked_sub(stack_size)
            .expect("stack region exceeds I/O region");

        // heap grows *up* from RAM_START_ADDRESS
        let memory_end = RAM_START_ADDRESS
            .checked_add(memory_size)
            .expect("memory_end overflow");

        Self {
            // program_size,
            // max_trusted_advice_size,
            max_untrusted_advice_size,
            max_input_size,
            max_output_size,
            // trusted_advice_start,
            // trusted_advice_end,
            untrusted_advice_start,
            untrusted_advice_end,
            input_start,
            input_end,
            output_start,
            output_end,
            stack_size,
            stack_end,
            memory_size,
            memory_end,
            panic,
            termination,
        }
    }
}
