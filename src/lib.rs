//! Ergonomics & safety focused deep learning in Rust. Main features include:
//! 1. Tensor library with shapes up to 6d!
//! 2. Shapes with both compile and runtime sized dimensions. (e.g. `Tensor<(usize, Const<10>)>` and `Tensor<Rank2<5, 10>>`)
//! 3. A large library of tensor operations (including `matmul`, `conv2d`, and much more).
//!     a. All tensor operations shape and type checked at compile time!!
//! 4. Ergonomic neural network building blocks (like `Linear`, `Conv2D`, and `Transformer`).
//! 5. Standard deep learning optimizers such as `Sgd`, `Adam`, `AdamW`, `RMSprop`, and more.
//! 6. Reverse mode auto differentiation implementation.
//! 7. Serialization to/from `.npy` and `.npz` for transferring models to/from python.
//!
//! # A quick tutorial
//!
//! 1. [crate::tensor::Tensor]s can be created with normal rust arrays. See [crate::tensor].
//! ```rust
//! # use dfdx::prelude::*;
//! let dev: Cpu = Default::default();
//! let x = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let y: Tensor<Rank2<2, 3>, f32, Cpu> = dev.ones();
//! // Runtime shape
//! let z: Tensor<(usize, Const<3>), f32, _> = dev.ones_like(&(10, Const));
//! ```
//!
//! 2. Neural networks are built with types. Tuples are sequential models. See [crate::nn].
//! ```rust
//! # use dfdx::prelude::*;
//! type Mlp = (
//!     Linear<5, 3>,
//!     ReLU,
//!     Linear<3, 2>,
//! );
//! ```
//!
//! 3. Instantiate models with [crate::nn::DeviceBuildExt]
//! ```rust
//! # use dfdx::prelude::*;
//! let dev: Cpu = Default::default();
//! type Model = (Linear<5, 2>, ReLU);
//! let mlp = dev.build_module::<Model, f32>();
//! ```
//!
//! 4. Pass data through networks with [crate::nn::Module]
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! # let mlp = dev.build_module::<Linear<5, 2>, f32>();
//! let x: Tensor<Rank1<5>, f32, _> = dev.zeros();
//! let y = mlp.forward(x); // compiler infers that `y` must be `Tensor<Rank1<2>>`
//! ```
//!
//! 5. Trace gradients using [crate::tensor::Trace::trace()]
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! # let mlp = dev.build_module::<Linear<10, 5>, f32>();
//! # let y_true: Tensor<Rank1<5>, f32, _> = dev.sample_normal().softmax();
//! // allocate gradients [ZeroGrads::alloc_grads]
//! let grads = mlp.alloc_grads();
//!
//! // tensors default to not having a tape
//! let x: Tensor<Rank1<10>, f32, Cpu, NoneTape> = dev.zeros();
//!
//! // `.trace()` clones `x` and inserts a gradient tape.
//! let x_traced: Tensor<Rank1<10>, f32, Cpu, OwnedTape<f32, Cpu>> = x.trace(grads);
//!
//! // The tape from the input is moved through the network during .forward().
//! let y: Tensor<Rank1<5>, f32, Cpu, NoneTape> = mlp.forward(x);
//! let y_traced: Tensor<Rank1<5>, f32, Cpu, OwnedTape<f32, Cpu>> = mlp.forward(x_traced);
//! ```
//!
//! 6. Compute gradients with [crate::tensor_ops::Backward]. See [crate::tensor_ops].
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! # let mlp = dev.build_module::<Linear<10, 5>, f32>();
//! # let y_true = dev.sample_normal::<Rank1<5>>().softmax();
//! # let y = mlp.forward(dev.zeros::<Rank1<10>>().trace(Gradients::leaky()));
//! // compute cross entropy loss
//! let loss = cross_entropy_with_logits_loss(y, y_true);
//!
//! // call `backward()` to compute gradients. The tensor *must* have `OwnedTape`!
//! let gradients: Gradients<f32, Cpu> = loss.backward();
//! ```
//! 7. Use an optimizer from [crate::optim] to optimize your network!
//! ```rust
//! # use dfdx::{prelude::*, optim::*};
//! # let dev: Cpu = Default::default();
//! # let mut mlp = dev.build_module::<Linear<10, 5>, f32>();
//! # let y_true = dev.sample_normal::<Rank1<5>>().softmax();
//! # let y = mlp.forward(dev.zeros::<Rank1<10>>().trace(Gradients::leaky()));
//! # let loss = cross_entropy_with_logits_loss(y, y_true);
//! # let mut gradients: Gradients<f32, Cpu> = loss.backward();
//! // Use stochastic gradient descent (Sgd), with a learning rate of 1e-2, and 0.9 momentum.
//! let mut opt = Sgd::new(&mlp, SgdConfig {
//!     lr: 1e-2,
//!     momentum: Some(Momentum::Classic(0.9)),
//!     weight_decay: None,
//! });
//!
//! // pass the gradients & the mlp into the optimizer's update method
//! opt.update(&mut mlp, &gradients);
//! mlp.zero_grads(&mut gradients);
//! ```

#![cfg_attr(feature = "no-std", no_std)]
#![allow(incomplete_features)]
#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

#[cfg(feature = "no-std")]
#[macro_use]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;

#[cfg(all(feature = "std", feature = "no-std"))]
compile_error!("Can't enable both std and no-std. Set default-features = false to disable std");

pub mod data;
pub mod feature_flags;
pub mod losses;
pub mod nn;
pub mod optim;
pub mod shapes;
pub mod tensor;
pub mod tensor_ops;

/// Contains subset of all public exports.
pub mod prelude {
    pub use crate::losses::*;
    pub use crate::nn::builders::*;
    pub use crate::optim::prelude::*;
    pub use crate::shapes::*;
    pub use crate::tensor::*;
    pub use crate::tensor_ops::*;
}

/// Sets a CPU `sse` flag to flush denormal floating point numbers to zero. The opposite of this is [keep_denormals()].
///
/// Some resources:
/// 1. [Effects of Flush-To-Zero mode](https://developer.arm.com/documentation/dui0473/c/neon-and-vfp-programming/the-effects-of-using-flush-to-zero-mode?lang=en)
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
/// 1. [Effects of Flush-To-Zero mode](https://developer.arm.com/documentation/dui0473/c/neon-and-vfp-programming/the-effects-of-using-flush-to-zero-mode?lang=en)
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

#[cfg(test)]
pub(crate) mod tests {

    #[cfg(not(feature = "test-cuda"))]
    pub type TestDevice = crate::tensor::Cpu;

    #[cfg(feature = "test-cuda")]
    pub type TestDevice = crate::tensor::Cuda;

    #[cfg(not(feature = "test-f64"))]
    pub type TestDtype = f32;

    #[cfg(feature = "test-f64")]
    pub type TestDtype = f64;

    pub trait AssertClose {
        type Elem: std::fmt::Display + std::fmt::Debug + Copy;
        const DEFAULT_TOLERANCE: Self::Elem;
        fn get_far_pair(
            &self,
            rhs: &Self,
            tolerance: Self::Elem,
        ) -> Option<(Self::Elem, Self::Elem)>;
        fn assert_close(&self, rhs: &Self, tolerance: Self::Elem)
        where
            Self: std::fmt::Debug,
        {
            if let Some((l, r)) = self.get_far_pair(rhs, tolerance) {
                panic!("lhs != rhs | {l} != {r}\n\n{self:?}\n\n{rhs:?}");
            }
        }
    }

    impl AssertClose for f32 {
        type Elem = f32;
        const DEFAULT_TOLERANCE: Self::Elem = 1e-6;
        fn get_far_pair(&self, rhs: &Self, tolerance: f32) -> Option<(f32, f32)> {
            if (self - rhs).abs() > tolerance {
                Some((*self, *rhs))
            } else {
                None
            }
        }
    }

    impl AssertClose for f64 {
        type Elem = f64;
        const DEFAULT_TOLERANCE: Self::Elem = 1e-6;
        fn get_far_pair(&self, rhs: &Self, tolerance: f64) -> Option<(f64, f64)> {
            if (self - rhs).abs() > tolerance {
                Some((*self, *rhs))
            } else {
                None
            }
        }
    }

    impl<T: AssertClose, const M: usize> AssertClose for [T; M] {
        type Elem = T::Elem;
        const DEFAULT_TOLERANCE: Self::Elem = T::DEFAULT_TOLERANCE;
        fn get_far_pair(
            &self,
            rhs: &Self,
            tolerance: Self::Elem,
        ) -> Option<(Self::Elem, Self::Elem)> {
            for (l, r) in self.iter().zip(rhs.iter()) {
                if let Some(pair) = l.get_far_pair(r, tolerance) {
                    return Some(pair);
                }
            }
            None
        }
    }

    pub fn assert_close<T: AssertClose + std::fmt::Debug>(a: &T, b: &T) {
        a.assert_close(b, T::DEFAULT_TOLERANCE);
    }

    pub fn assert_close_with_tolerance<T: AssertClose + std::fmt::Debug>(
        a: &T,
        b: &T,
        tolerance: T::Elem,
    ) {
        a.assert_close(b, tolerance);
    }
}
