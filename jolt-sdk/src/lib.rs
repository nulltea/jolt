#![cfg_attr(not(feature = "host"), no_std)]

extern crate jolt_sdk_macros;

pub use jolt_sdk_macros::provable;
pub use postcard;

#[cfg(feature = "host")]
pub mod host_utils;
#[cfg(feature = "host")]
pub use host_utils::*;

pub mod alloc;
pub use alloc::*;

use serde::{Deserialize, Serialize};

/// A wrapper type to mark guest program inputs as untrusted_advice.
#[derive(Debug, Serialize, Deserialize)]
#[repr(transparent)]
pub struct UntrustedAdvice<T> {
    value: T,
}

impl<T> UntrustedAdvice<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }

    pub fn unwrap(self) -> T {
        self.value
    }
}

impl<T> From<T> for UntrustedAdvice<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> core::ops::Deref for UntrustedAdvice<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

// This is a dummy _HEAP_PTR to keep the compiler happy.
// It should never be used when compiled as a guest or with
// our custom allocator
#[no_mangle]
#[cfg(feature = "host")]
pub static mut _HEAP_PTR: u8 = 0;
