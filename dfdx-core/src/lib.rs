//! # dfdx
//!
//! dfdx is a cuda accelerated tensor and neural network library, writtten
//! entirely in rust!
//!
//! Additionally, it can track compile time shapes across tensor operations,
//! ensuring that all your neural networks are checked **at compile time**.
//!
//! The following sections provide some high level core concepts & exmaples, and
//! there is more detailed documentation in each of dfdx's submodules.
//!
//! See [feature_flags] for details on feature flags.
//!
//! # Shapes & Tensors
//!
//! *See [dtypes], [shapes], and [tensor] for more information.*
//!
//! At its core a [`tensor::Tensor`] is just a nd-array. Just like
//! rust arrays there are two parts:
//! 1. Shape ([shapes])
//! 2. Dtype ([dtypes])
//!
//! dfdx represents shapes as **tuples** of dimensions ([`shapes::Dim`]),
//! where a dimension can either be known at:
//! 1. Compile time [`shapes::Const<M>`]
//! 2. Run time [`usize`]
//!
//! You can freely mix and match these dimensions together. Here are some
//! example shapes:
//! - `()` - unit shape
//! - `(usize,)` - 1d shape with a runtime known dimension
//! - `(usize, Const<5>)` - 2d shape with both types of dimensions
//! - `(Const<3>, usize, Const<5>)` - 3d shape!
//! - `Rank3<3, 5, 7>` - Equivalent to `(Const<3>, Const<5>, Const<7>)`
//!
//! Here are some comparisons between representing nd arrays in rust vs dfdx:
//!
//! | rust array | dfdx `Tensor` |
//! | --- | --- |
//! | f32 | Tensor<(), f32, ...> |
//! | [u32; 5] | Tensor<Rank1<5>, u32, ...> |
//! | [[u8; 3]; 2] | Tensor<Rank2<2, 3>, u8, ...> |
//! | Vec<[bool; 5]> | Tensor<(usize, Const<5>), bool, ...> |
//!
//! The `Rank1`, `Rank2` shapes used above are actually type aliases for
//! when **all dimensions are compile time**:
//! - [`shapes::Rank0`] is just `()`.
//! - [`shapes::Rank1<M>`] is `(Const<M>, )`
//! - [`shapes::Rank2<M, N>`] is `(Const<M>, Const<N>)`
//!
//! # Allocating tensors with Devices
//!
//! *See [tensor] for more information.*
//!
//! Devices are used to allocate tensors (and neural networks!). They are akin
//! to [std::alloc::GlobalAlloc] in rust - they just allocate memory.
//! They are also used to execute tensor ops, which we will get to later on.
//!
//! There are two options for this currently, with more planned to be added in the future:
//!
//! 1. [tensor::Cpu] - for tensors stored on the heap
//! 2. [tensor::Cuda] - for tensors stored in GPU memory
//!
//! Both devices implement [Default], you can also create them with a certain seed
//! and ordinal.
//!
//! Here's how you might use a device:
//!
//! ```rust
//! # use dfdx_core::prelude::*;
//! let dev: Cpu = Default::default();
//! let t: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
//! ```
//!
//! # Tensor Operations (tip of the iceberg)
//!
//! *See [tensor_ops] for more information*
//!
//! Once you've instantiated tensors with a device, you can start doing operations on them!
//! There are **many many** operations, here are a few core ones and how they related
//! to things like numpy/pytorch:
//!
//! | Operation | dfdx | numpy | pytorch |
//! | --- | --- | --- | --- |
//! | Unary Operations | `a.sqrt()` | `a.sqrt()` | `a.sqrt()` |
//! | Binary Operations | `a + b` | `a + b` | `a + b` |
//! | gemm/gemv | [tensor_ops::matmul] | `a @ b` | `a @ b` |
//! | 2d Convolution | [tensor_ops::TryConv2D] | - | `torch.conv2d` |
//! | 2d Transposed Convolution | [tensor_ops::TryConvTrans2D] | - | `torch.conv_transpose2d` |
//! | Slicing | [tensor_ops::slice] | `a[...]` | `a[...]` |
//! | Select | [tensor_ops::SelectTo] | `a[...]` | `torch.select` |
//! | Gather | [tensor_ops::GatherTo] | `np.take` | `torch.gather` |
//! | Broadcasting | [tensor_ops::BroadcastTo] | implicit/`np.broadcast` | implicit/`torch.broadcast_to` |
//! | Permute | [tensor_ops::PermuteTo] | `np.transpose(...)` | `torch.permute` |
//! | Where | [tensor_ops::ChooseFrom] | `np.where` | `torch.where` |
//! | Reshape | [tensor_ops::ReshapeTo] | `np.reshape(shape)` | `a.reshape(shape)` |
//! | View | [tensor_ops::ReshapeTo] | `np.view(...)` | `a.view(...)` |
//! | Roll | [tensor_ops::Roll] | `np.rollaxis(...)` | `a.roll(...)` |
//! | Stack | [tensor_ops::TryStack] | `np.stack` | `torch.stack` |
//! | Concat | [tensor_ops::TryConcat] | `np.concatenate` | `torch.concat` |
//!
//! and **much much more!**

#![cfg_attr(all(feature = "no-std", not(feature = "std")), no_std)]
#![allow(incomplete_features)]
#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

#[cfg(feature = "no-std")]
#[macro_use]
extern crate alloc;
#[cfg(all(feature = "no-std", not(feature = "std")))]
extern crate no_std_compat as std;
extern crate self as dfdx_core;

pub mod data;
pub mod dtypes;
pub mod losses;
pub mod nn_traits;
pub mod shapes;
pub mod tensor;
pub mod tensor_ops;

/// Contains subset of all public exports.
pub mod prelude {
    pub use crate::losses::*;
    pub use crate::shapes::*;
    pub use crate::tensor::*;
    pub use crate::tensor_ops::*;
}

#[cfg(test)]
pub(crate) mod tests {
    pub use num_traits::{Float, NumCast, Zero};

    #[cfg(not(feature = "cuda"))]
    pub type TestDevice = crate::tensor::Cpu;

    #[cfg(feature = "cuda")]
    pub type TestDevice = crate::tensor::Cuda;

    #[cfg(all(feature = "test-f64", feature = "test-f16"))]
    compile_error!("f64 and f16 cannot be tested at the same time");

    #[cfg(all(
        not(feature = "test-amp-f16"),
        not(feature = "test-f16"),
        not(feature = "test-f64")
    ))]
    pub type TestDtype = f32;

    #[cfg(feature = "test-f16")]
    pub type TestDtype = half::f16;

    #[cfg(feature = "test-f64")]
    pub type TestDtype = f64;

    #[cfg(feature = "test-amp-f16")]
    pub type TestDtype = crate::dtypes::AMP<half::f16>;

    pub trait AssertClose {
        type Elem: std::fmt::Display + std::fmt::Debug + Copy;
        const DEFAULT_TOLERANCE: Self::Elem;
        fn get_default_tol(&self) -> Self::Elem {
            Self::DEFAULT_TOLERANCE
        }
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

    impl<F: Copy + std::fmt::Debug + std::fmt::Display + AssertClose> AssertClose
        for crate::dtypes::AMP<F>
    {
        type Elem = crate::dtypes::AMP<F::Elem>;
        const DEFAULT_TOLERANCE: Self::Elem = crate::dtypes::AMP(F::DEFAULT_TOLERANCE);
        fn get_far_pair(
            &self,
            rhs: &Self,
            tolerance: Self::Elem,
        ) -> Option<(Self::Elem, Self::Elem)> {
            self.0
                .get_far_pair(&rhs.0, tolerance.0)
                .map(|(l, r)| (crate::dtypes::AMP(l), crate::dtypes::AMP(r)))
        }
    }

    #[cfg(feature = "f16")]
    impl AssertClose for half::f16 {
        type Elem = Self;
        const DEFAULT_TOLERANCE: Self::Elem = half::f16::from_f32_const(1e-2);
        fn get_far_pair(&self, rhs: &Self, tolerance: Self) -> Option<(Self, Self)> {
            if num_traits::Float::abs(self - rhs) > tolerance {
                Some((*self, *rhs))
            } else {
                None
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

    pub trait NdMap {
        type Elem;
        type Mapped<O>;
        fn ndmap<O, F: Copy + FnMut(Self::Elem) -> O>(self, f: F) -> Self::Mapped<O>;
    }

    impl NdMap for f32 {
        type Elem = Self;
        type Mapped<O> = O;
        fn ndmap<O, F: Copy + FnMut(Self::Elem) -> O>(self, mut f: F) -> O {
            f(self)
        }
    }

    impl NdMap for f64 {
        type Elem = Self;
        type Mapped<O> = O;
        fn ndmap<O, F: Copy + FnMut(Self::Elem) -> O>(self, mut f: F) -> O {
            f(self)
        }
    }

    impl<T: NdMap, const M: usize> NdMap for [T; M] {
        type Elem = T::Elem;
        type Mapped<O> = [T::Mapped<O>; M];
        fn ndmap<O, F: Copy + FnMut(Self::Elem) -> O>(self, f: F) -> Self::Mapped<O> {
            self.map(|t| t.ndmap(f))
        }
    }

    macro_rules! assert_close_to_literal {
        ($Lhs:expr, $Rhs:expr) => {{
            let lhs = $Lhs.array();
            let rhs = $Rhs.ndmap(|x| num_traits::FromPrimitive::from_f64(x).unwrap());
            let tol = AssertClose::get_default_tol(&lhs);
            let far_pair = AssertClose::get_far_pair(&lhs, &rhs, tol);
            if let Some((l, r)) = far_pair {
                panic!("lhs != rhs | {l} != {r}");
            }
        }};
        ($Lhs:expr, $Rhs:expr, $Tolerance:expr) => {{
            let far_pair = $Lhs.array().get_far_pair(
                &$Rhs.ndmap(|x| num_traits::FromPrimitive::from_f64(x).unwrap()),
                num_traits::FromPrimitive::from_f64($Tolerance).unwrap(),
            );
            if let Some((l, r)) = far_pair {
                panic!("lhs != rhs | {l} != {r}");
            }
        }};
    }
    pub(crate) use assert_close_to_literal;

    macro_rules! assert_close_to_tensor {
        ($Lhs:expr, $Rhs:expr) => {
            let lhs = $Lhs.array();
            let tol = AssertClose::get_default_tol(&lhs);
            let far_pair = AssertClose::get_far_pair(&lhs, &$Rhs.array(), tol);
            if let Some((l, r)) = far_pair {
                panic!("lhs != rhs | {l} != {r}");
            }
        };
        ($Lhs:expr, $Rhs:expr, $Tolerance:expr) => {{
            let far_pair = $Lhs.array().get_far_pair(
                &$Rhs.array(),
                num_traits::FromPrimitive::from_f64($Tolerance).unwrap(),
            );
            if let Some((l, r)) = far_pair {
                panic!("lhs != rhs | {l} != {r}");
            }
        }};
    }
    pub(crate) use assert_close_to_tensor;

    macro_rules! assert_close {
        ($Lhs:expr, $Rhs:expr) => {
            let lhs = $Lhs;
            let tol = AssertClose::get_default_tol(&lhs);
            let far_pair = AssertClose::get_far_pair(&lhs, &$Rhs, tol);
            if let Some((l, r)) = far_pair {
                panic!("lhs != rhs | {l} != {r}");
            }
        };
        ($Lhs:expr, $Rhs:expr, $Tolerance:expr) => {{
            let far_pair = $Lhs.get_far_pair(
                &$Rhs,
                num_traits::FromPrimitive::from_f64($Tolerance).unwrap(),
            );
            if let Some((l, r)) = far_pair {
                panic!("lhs != rhs | {l} != {r}");
            }
        }};
    }

    pub(crate) use assert_close;
}
