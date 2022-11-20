//! Operations on tensors like [relu()], [matmul()], [softmax()], and more.
//!
//! # Generic function and struct methods
//!
//! All functionality is provided in two ways.
//! 1. The generic standalone function that takes a generic parameter. e.g. [relu()].
//! 2. The struct method for tensor structs. e.g. [crate::tensor::Tensor1D::relu()].
//!
//! The struct methods are all just pass throughs to the generic function.
//!
//! # Axes/Dimensions for broadcasting/reductions/selecting
//!
//! For the following sections, some traits/functions utilizing const `isize` to determine
//! the axis to apply the transformation to.
//!
//! Here are the valid axes for each tensor:
//! 1. `Tensor0D`: `Axis<0>`
//! 2. `Tensor1D`: `Axis<0>`
//! 3. `Tensor2D`: `Axis<0>`, `Axis<1>`
//! 4. `Tensor3D`: `Axis<0>`, `Axis<1>`, `Axis<2>`,
//! 5. `Tensor4D`: `Axis<0>`, `Axis<1>`, `Axis<2>`, `Axis<3>`
//!
//! Additionally `AllAxes` is valid for all tensors.
//! To specify multiple axes you can use `Axes2`, `Axes3`, and `Axes4`
//!
//! # Reductions
//!
//! There are a number of functions that reduce 1 or more axes. Valid axes and reductions
//! can be seen by viewing the [Reduce] or [ReduceTo] traits. Anything that can be [Reduce]'d can also
//! be [BroadcastTo] the same tensor.
//!
//! There are 2 ways to call each axis reducing function:
//! 1. The tensor method (e.g. [crate::tensor::Tensor1D::sum()]), where the axes are inferred based
//!    on the output type.
//! ```rust
//! # use dfdx::prelude::*;
//! let t: Tensor3D<2, 4, 6> = TensorCreator::zeros();
//! let _: Tensor1D<4> = t.sum();
//! ```
//! 2. The generic function (e.g. [sum]), where you need to specify the axes as generic parameters
//! ```rust
//! # use dfdx::prelude::*;
//! let t: Tensor3D<2, 4, 6> = TensorCreator::zeros();
//! let _: Tensor1D<4> = sum::<_, Axes2<0, 2>>(t);
//! ```
//!
//! Complete list of reductions:
//!
//! - [max()]
//! - [mean()]
//! - [min()]
//! - [sum()]
//! - [var()]
//! - [stddev()]
//! - [logsumexp()]
//!
//! # Broadcasts
//!
//! Broadcasting tensors is provided through the [BroadcastTo] trait. Generally the axes
//! can be inferred by the type of the output, so you don't have to explicitly
//! specify them.
//!
//! To broadcast a tensor to be the same size as another tensor you can use like so:
//! ```rust
//! # use dfdx::prelude::*;
//! let big: Tensor2D<2, 5> = TensorCreator::zeros();
//!
//! // broadcast the 1nd axis
//! let a: Tensor2D<2, 5> = Tensor1D::<5>::zeros().broadcast();
//! add(a, big.clone());
//!
//!// broadcast the 2nd axis
//! let a: Tensor2D<2, 5> = Tensor1D::<2>::zeros().broadcast();
//! add(a, big);
//! ```
//!
//! # Permutating axes
//!
//! Permutating axes is done via [PermuteTo], and similar to braodcasting/reducing,
//! you can just specify the output type and the axes will be inferred.
//!
//! 2D version:
//! ```rust
//! # use dfdx::prelude::*;
//! let t: Tensor2D<2, 3> = TensorCreator::zeros();
//! let _: Tensor2D<3, 2> = t.permute();
//! ```
//!
//! 3D version:
//! ```rust
//! # use dfdx::prelude::*;
//! let t: Tensor3D<2, 3, 4> = TensorCreator::zeros();
//! let _: Tensor3D<3, 4, 2> = t.permute();
//! ```
//!
//! 4D version:
//! ```rust
//! # use dfdx::prelude::*;
//! let t: Tensor4D<2, 3, 4, 5> = TensorCreator::zeros();
//! let _: Tensor4D<3, 5, 2, 4> = t.permute();
//! ```
//!
//! # Selects/Indexing
//!
//! Selecting or indexing into a tensor is done via [SelectTo::select()]. This traits enables
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
mod impl_add;
mod impl_backward;
mod impl_broadcast_reduce;
// mod impl_clamp;
// mod impl_div;
// mod impl_dropout;
// mod impl_mask;
// mod impl_max;
// mod impl_maximum;
mod impl_mean;
// mod impl_min;
// mod impl_minimum;
// mod impl_mul;
// mod impl_nans;
// mod impl_normalize;
// mod impl_pow;
// mod impl_softmax;
// mod impl_stddev;
mod impl_sub;
mod impl_sum;
mod map;
// mod matmul;
mod permute;
// mod select;
pub(crate) mod utils;

// pub use arith_scalar::*;
// pub use impl_add::*;
// pub use impl_backward::*;
// pub use impl_broadcast_reduce::*;
// pub use impl_clamp::*;
// pub use impl_div::*;
// pub use impl_dropout::*;
// pub use impl_mask::*;
// pub use impl_max::*;
// pub use impl_maximum::*;
// pub use impl_mean::*;
// pub use impl_min::*;
// pub use impl_minimum::*;
// pub use impl_mul::*;
// pub use impl_nans::*;
// pub use impl_normalize::*;
// pub use impl_pow::*;
// pub use impl_softmax::*;
// pub use impl_stddev::*;
// pub use impl_sub::*;
// pub use impl_sum::*;
// pub use map::*;
// pub use matmul::*;
// pub use permute::*;
// pub use select::SelectTo;

// #[cfg(feature = "nightly")]
// mod impl_reshape;
// #[cfg(feature = "nightly")]
// pub use impl_reshape::*;

// #[cfg(feature = "nightly")]
// mod conv;
// #[cfg(feature = "nightly")]
// pub use conv::*;

// #[cfg(feature = "nightly")]
// mod pool2d;
// #[cfg(feature = "nightly")]
// pub use pool2d::*;
