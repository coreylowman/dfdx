//! Ergonomics & safety focused deep learning in Rust. Main features include:
//! 1. Tensor library, complete with const generic shapes, activation functions, and more.
//! 2. Safe & Easy to use neural network building blocks.
//! 3. Standard deep learning optimizers such as Sgd and Adam.
//! 4. Reverse mode auto differentiation[1] implementation.

pub mod arrays;
pub mod data;
pub mod devices;
pub mod gradients;
pub mod losses;
pub mod nn;
pub mod numpy;
pub mod optim;
pub mod tensor;
pub mod tensor_ops;
pub mod unique_id;

/// Contains all public exports.
pub mod prelude {
    pub use crate::arrays::*;
    pub use crate::data::*;
    pub use crate::devices::*;
    pub use crate::gradients::*;
    pub use crate::losses::*;
    pub use crate::nn::*;
    pub use crate::optim::*;
    pub use crate::tensor::*;
    pub use crate::tensor_ops::*;
    pub use crate::unique_id::*;
}

/// Sets a CPU `sse` flag to flush denormal floating point numbers to zero. The opposite of this is [keep_denormals()].
///
/// Some resources:
/// 1. [Effects of Flush-To-Zero mode]https://developer.arm.com/documentation/dui0473/c/neon-and-vfp-programming/the-effects-of-using-flush-to-zero-mode?lang=en
/// 2. [When to use Flush-To-Zero mode](https://developer.arm.com/documentation/dui0473/c/neon-and-vfp-programming/when-to-use-flush-to-zero-mode?lang=en)
pub fn flush_denormals_to_zero() {
    #[cfg(all(target_arch = "x86", target_feature = "sse"))]
    {
        use std::arch::x86::{_MM_FLUSH_ZERO_ON, _MM_SET_FLUSH_ZERO_MODE};
        unsafe { _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON) }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    {
        use std::arch::x86_64::{_MM_FLUSH_ZERO_ON, _MM_SET_FLUSH_ZERO_MODE};
        unsafe { _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON) }
    }
}

/// Sets a CPU flag to keep denormal floating point numbers. The opposite of this is [flush_denormals_to_zero()].
///
/// Some resources:
/// 1. [Effects of Flush-To-Zero mode]https://developer.arm.com/documentation/dui0473/c/neon-and-vfp-programming/the-effects-of-using-flush-to-zero-mode?lang=en
/// 2. [When to use Flush-To-Zero mode](https://developer.arm.com/documentation/dui0473/c/neon-and-vfp-programming/when-to-use-flush-to-zero-mode?lang=en)
pub fn keep_denormals() {
    #[cfg(all(target_arch = "x86", target_feature = "sse"))]
    {
        use std::arch::x86::{_MM_FLUSH_ZERO_OFF, _MM_SET_FLUSH_ZERO_MODE};
        unsafe { _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF) }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    {
        use std::arch::x86_64::{_MM_FLUSH_ZERO_OFF, _MM_SET_FLUSH_ZERO_MODE};
        unsafe { _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF) }
    }
}
