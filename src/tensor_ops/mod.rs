//! Implementations of all ops for tensors including activations like [relu()], binary operations like [matmul()], and more.
//!
//! # Generic function and struct methods
//!
//! All functionality is provided in two ways.
//! 1. The generic standalone function that takes a generic parameter. e.g. [mean()].
//! 2. The struct method for tensor structs. e.g. [crate::prelude::Tensor1D::mean()].
//!
//! The struct methods are all just pass throughs to the generic function.
//!
//! # Reductions
//!
//! There are a number of functions that reduce a dimension (e.g. [mean_axis()]).
//! These functions are all labeled with `*_axis()` at the end.
//!
//! Reducing a dimension means removing that dimension from the tensor by reducing it to 1 number.
//! For example calling [sum_axis::<1>()] on a `Tensor2D<2, 5>` would result in a `Tensor1D<2>`.
//! Calling [sum_axis::<0>()] on a `Tensor2D<5>` would result in a `Tensor1D<5>`.
//!
//! See relevant functions for more examples.
//!
//! # Broadcasts
//!
//! Broadcasting tensors is provided through the following traits:
//! 1. [Broadcast1], which broadcasts a single axis
//! 2. [Broadcast2], which broadcasts 2 axes
//! 3. [Broadcast3], which broadcasts 3 axes
//! 4. [Broadcast4], which broadcasts 4 axes
//!
//! To broadcast a tensor to be the same size as another tensor you can use like so:
//! ```rust
//! # use dfdx::prelude::*;
//! let a: Tensor2D<2, 5> = TensorCreator::zeros();
//! let b: Tensor1D<2> = TensorCreator::zeros();
//! let c = b.broadcast1::<1>();
//! add(a, &c);
//! ```
//! or
//! ```rust
//! # use dfdx::prelude::*;
//! let a: Tensor2D<2, 5> = TensorCreator::zeros();
//! let b: Tensor1D<5> = TensorCreator::zeros();
//! let c = b.broadcast1::<0>();
//! add(a, &c);
//! ```

mod arith;
mod arith_scalar;
pub(super) mod binary_map;
mod broadcast;
mod impl_backward;
mod impl_clamp;
mod impl_dropout;
mod impl_gather_last;
mod impl_mask;
mod impl_max_axis;
mod impl_mean;
mod impl_mean_axis;
mod impl_nans;
mod impl_normalize;
mod impl_softmax;
mod impl_std_axis;
mod impl_sum;
mod impl_sum_axis;
mod map;
mod matmul;
mod reduce;
mod utils;

pub use arith::*;
pub use arith_scalar::*;
pub use broadcast::*;
pub use impl_backward::*;
pub use impl_clamp::*;
pub use impl_dropout::*;
pub use impl_gather_last::*;
pub use impl_mask::*;
pub use impl_max_axis::*;
pub use impl_mean::*;
pub use impl_mean_axis::*;
pub use impl_nans::*;
pub use impl_normalize::*;
pub use impl_softmax::*;
pub use impl_std_axis::*;
pub use impl_sum::*;
pub use impl_sum_axis::*;
pub use map::*;
pub use matmul::*;
pub use reduce::*;

#[cfg(feature = "nightly")]
mod impl_reshape;
#[cfg(feature = "nightly")]
pub use impl_reshape::*;
