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
//! There are a number of functions that reduce a dimension (e.g. [mean_last_dim()]).
//! These functions are all labeled with `*_last_dim()` at the end.
//!
//! Reducing a dimension means removing that dimension from the tensor by reducing it to 1 number.
//! For example calling [sum_last_dim()] on a `Tensor2D<2, 5>` would result in a `Tensor1D<2>`.
//!
//! See relevant functions for more examples.
//!
//! # Broadcasts
//!
//! Some binary functions need to broadcast one argument to be the same size as the other (e.g. [add_broadcast_rhs_last()]).
//! These methods are named as `<operation>_broadcast_<argument>_<dimension>()`. Currently all of the functions
//! broadcast the second argument (`rhs`). And there are version where the first dimension is broadcast and the last dimension
//! is broadcast
//!
//! 1. [add_broadcast_rhs_last()] (and others) broadcasts the last dimension
//! 2. [add_broadcast_rhs_first()] (and others) broadcast the entire array according to the first dimension in `lhs`.
//!
//! See relevant functions for more examples.

mod arith;
mod arith_broadcast_inner;
mod arith_broadcast_outer;
mod arith_scalar;
pub(super) mod binary_map;
mod impl_activations;
mod impl_backward;
mod impl_clamp;
mod impl_dropout;
mod impl_gather_last;
mod impl_mask;
mod impl_max_last;
mod impl_mean;
mod impl_mean_last;
mod impl_nans;
mod impl_neg;
mod impl_normalize;
mod impl_softmax;
mod impl_std_last;
mod impl_sum;
mod impl_sum_last;
mod matmul;
mod reshape;

pub use arith::*;
pub use arith_broadcast_inner::*;
pub use arith_broadcast_outer::*;
pub use arith_scalar::*;
pub use impl_activations::*;
pub use impl_backward::*;
pub use impl_clamp::*;
pub use impl_dropout::*;
pub use impl_gather_last::*;
pub use impl_mask::*;
pub use impl_max_last::*;
pub use impl_mean::*;
pub use impl_mean_last::*;
pub use impl_nans::*;
pub use impl_neg::*;
pub use impl_normalize::*;
pub use impl_softmax::*;
pub use impl_std_last::*;
pub use impl_sum::*;
pub use impl_sum_last::*;
pub use matmul::*;
