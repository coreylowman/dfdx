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
//! # Axes/Dimensions for broadcasting/reductions/selecting
//!
//! For the following sections, some traits/functions utilizing const `isize` to determine
//! the axis to apply the transformation to.
//!
//! Here are the valid axes for each tensor:
//! 1. `Tensor0D`: `-1`
//! 2. `Tensor1D`: `0`, `-1`
//! 3. `Tensor2D`: `0`, `1`, `-1`
//! 4. `Tensor3D`: `0`, `1`, `2`, `-1`
//! 5. `Tensor4D`: `0`, `1`, `2`, `3`, `-1`
//!
//! Note that `-1` must be used for the last axis instead of the positive number. This is to prevent
//! ambiguous operations.
//!
//! # Reductions
//!
//! There are a number of functions that reduce a dimension (e.g. [mean_axis()]).
//! These functions are all labeled with `*_axis()` at the end.
//!
//! Reducing a dimension means removing that dimension from the tensor by reducing it to 1 number.
//! For example calling `sum_axis::<-1>()` on a `Tensor2D<2, 5>` would result in a `Tensor1D<2>`.
//! Calling [sum_axis::<0>()] on a `Tensor2D<5>` would result in a `Tensor1D<5>`.
//!
//! See [Reduce1] implementations for a complete list of reductions.
//!
//! See relevant functions for more examples.
//!
//! # Broadcasts
//!
//! Broadcasting tensors is provided through the following traits:
//! 1. [Broadcast1::broadcast1()], which broadcasts a single axis
//! 2. [Broadcast2::broadcast2()], which broadcasts 2 axes
//! 3. [Broadcast3::broadcast3()], which broadcasts 3 axes
//! 4. [Broadcast4::broadcast4()], which broadcasts 4 axes
//!
//! See the implementations of each trait for a complete list of possible broadcasts.
//!
//! To broadcast a tensor to be the same size as another tensor you can use like so:
//! ```rust
//! # use dfdx::prelude::*;
//! let big: Tensor2D<2, 5> = TensorCreator::zeros();
//!
//! // broadcast the 1nd axis
//! let a: Tensor2D<2, 5> = Tensor1D::<5>::zeros().broadcast1();
//! add(a, &big);
//!
//!// broadcast the 2nd axis
//! let a: Tensor2D<2, 5> = Tensor1D::<2>::zeros().broadcast1();
//! add(a, &big);
//! ```
//!
//! # Selects/Indexing
//!
//! Selecting or indexing into a tensor is done via [Select1::select()]. This traits enables
//! 2 behaviors for each axis of a given tensor:
//!
//! 1. Select exactly 1 element from that axis.
//! 2. Select Z elements (can be different from the size of the axis) from that axis
//!
//! For example here is selecting from the 0th axis of a 2d tensor:
//! ```rust
//! # use dfdx::prelude::*;
//! let t = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//!
//! let a: Tensor1D<3> = t.clone().select(&0); // select the first row
//! assert_eq!(a.data(), &[1.0, 2.0, 3.0]);
//!
//! let b: Tensor2D<5, 3> = t.select(&[0, 0, 1, 1, 1]); // select each row multiple times
//! ```
//!
//! This can be done per axis as well, which allows a number of combinations.
//! Here is the same example but selecting from the last axis of a 2d tensor:
//! ```rust
//! # use dfdx::prelude::*;
//! let t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//!
//! let a: Tensor1D<2> = t.clone().select(&[0, 2]); // select one element from the last axis
//! assert_eq!(a.data(), &[1.0, 6.0]);
//!
//! let b: Tensor2D<2, 2> = t.select(&[[0, 2], [1, 1]]); // select multiple from the last axis
//! assert_eq!(b.data(), &[[1.0, 3.0], [5.0, 5.0]]);
//! ```

mod arith_scalar;
pub mod binary_map;
mod broadcast;
mod impl_backward;
mod impl_clamp;
mod impl_dropout;
mod impl_mask;
mod impl_max_axis;
mod impl_mean;
mod impl_mean_axis;
mod impl_min_axis;
mod impl_nans;
mod impl_normalize_axis;
mod impl_softmax;
mod impl_std_axis;
mod impl_sum;
mod impl_sum_axis;
mod map;
mod matmul;
mod reduce;
mod select;
mod utils;

pub use arith_scalar::*;
pub use binary_map::*;
pub use broadcast::*;
pub use impl_backward::*;
pub use impl_clamp::*;
pub use impl_dropout::*;
pub use impl_mask::*;
pub use impl_max_axis::*;
pub use impl_mean::*;
pub use impl_mean_axis::*;
pub use impl_min_axis::*;
pub use impl_nans::*;
pub use impl_normalize_axis::*;
pub use impl_softmax::*;
pub use impl_std_axis::*;
pub use impl_sum::*;
pub use impl_sum_axis::*;
pub use map::*;
pub use matmul::*;
pub use reduce::*;
pub use select::*;

#[cfg(feature = "nightly")]
mod impl_reshape;
#[cfg(feature = "nightly")]
pub use impl_reshape::*;

#[cfg(feature = "nightly")]
mod conv;
#[cfg(feature = "nightly")]
pub use conv::*;
