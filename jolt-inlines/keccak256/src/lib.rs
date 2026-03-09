//! Keccak-256 inline.

#![cfg_attr(not(feature = "host"), no_std)]

pub const INLINE_OPCODE: u32 = 0x0B;

pub const KECCAK256_FUNCT3: u32 = 0x00;
pub const KECCAK256_FUNCT7: u32 = 0x01;
pub const KECCAK256_NAME: &str = "KECCAK256_INLINE";

pub const NUM_LANES: usize = 25;
pub type Keccak256State = [u64; NUM_LANES];

pub mod sdk;
pub use sdk::*;

#[cfg(all(feature = "host", feature = "rv64"))]
pub mod exec;
#[cfg(all(feature = "host", feature = "rv64"))]
pub mod sequence_builder;

#[cfg(all(feature = "host", feature = "rv64"))]
mod host;
#[cfg(all(feature = "host", feature = "rv64"))]
pub use host::*;

#[cfg(all(test, feature = "host"))]
pub mod test_constants;
#[cfg(all(test, feature = "host"))]
pub mod test_utils;
#[cfg(all(test, feature = "host"))]
mod tests;
