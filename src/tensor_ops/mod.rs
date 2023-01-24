//! Operations on tensors like [relu()], [matmul()], [softmax()], and more.
//!
//! # Generic function and struct methods
//!
//! All functionality is provided in two ways.
//! 1. The generic standalone function that takes a generic parameter. e.g. [relu()].
//! 2. The struct method for tensor structs. e.g. [crate::tensor::Tensor::relu()].
//!
//! The functions are all just pass throughs to the tensor methods.
//!
//! # Axes/Dimensions for broadcasting/reductions/selecting
//!
//! For the following sections, some traits/functions utilizing const `isize` to determine
//! the axis to apply the transformation to.
//!
//! Here are the valid axes for each tensor:
//! 1. 0d tensor: `Axis<0>`
//! 2. 1d tensor: `Axis<0>`
//! 3. 2d tensor: `Axis<0>`, `Axis<1>`
//! 4. 3d tensor: `Axis<0>`, `Axis<1>`, `Axis<2>`,
//! 5. 4d tensor: `Axis<0>`, `Axis<1>`, `Axis<2>`, `Axis<3>`
//! 6. etc.
//!
//! To specify multiple axes you can use `Axes2`, `Axes3`, and `Axes4`
//!
//! # Reductions
//!
//! There are a number of methods that reduce 1 or more axes.Anything that can be reduced can also
//! be broadcasted back to the original shape using [BroadcastTo].
//!
//! Each axis reducing function has two generic parameters:
//! 1. The target shape
//! 2. The axes to reduce along
//! **You only need to specify one of these!** Generally it is better practice to specify the
//! target shape, unless it is ambiguous in which case you should specify the axes.
//!
//! For example:
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t: Tensor<Rank3<2, 4, 6>> = dev.zeros();
//! // shape version
//! let _: Tensor<Rank1<4>> = t.clone().sum::<Rank1<4>, _>();
//! // axes version
//! let _: Tensor<Rank1<2>> = t.clone().sum::<_, Axes2<1, 2>>();
//! ```
//!
//! Complete list of reductions:
//!
//! - [MaxTo]
//! - [MeanTo]
//! - [MinTo]
//! - [SumTo]
//! - [VarTo]
//! - [StddevTo]
//! - [LogSumExpTo]
//!
//! # Broadcasts
//!
//! Broadcasting tensors is provided through the [BroadcastTo] trait. Similar to reductions
//! there are two generic parameters to broadcast:
//! 1. (Required) The target shape
//! 2. (usually optional) The axes *of the result type* to broadcast
//! You'll only need to specify axes if the shape makes the broadcasts ambiguous.
//!
//! For example:
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t: Tensor<Rank1<4>> = dev.zeros();
//! // shape version
//! let _: Tensor<Rank3<2, 4, 6>> = t.clone().broadcast::<Rank3<2, 4, 6>, _>();
//! ```
//!
//! Rust can also infer the output type if you use it in another operation:
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let big: Tensor<Rank2<2, 5>> = dev.zeros();
//! let small: Tensor<Rank1<5>> = dev.zeros();
//! let _ = big + small.broadcast();
//! ```
//!
//! # Permutes
//!
//! Permuting has an identical interface to broadcasts/reductions:
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t: Tensor<Rank3<2, 3, 4>> = dev.zeros();
//! // shape version
//! let _ = t.clone().permute::<Rank3<3, 4, 2>, _>();
//! // axes version
//! let _ = t.clone().permute::<_, Axes3<1, 2, 0>>();
//! ```
//!
//! # Indexing using select and gather
//!
//! Two traits provide indexing capability [SelectTo] and [GatherTo]. The difference is:
//! 1. [SelectTo::select] allows you to select a single value
//! 2. [GatherTo::gather] allows you select multiple values from the same axis.
//!
//! For example you can select from the 0th axis like so:
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let r: Tensor<Rank1<3>> = t.select(dev.tensor(1));
//! assert_eq!(r.array(), [4.0, 5.0, 6.0]);
//! ```
//!
//! Or you can gather from the 0th axis to select multiple entries:
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let r: Tensor<Rank2<3, 3>> = t.gather(dev.tensor([1, 1, 0]));
//! assert_eq!(r.array(), [
//!     [4.0, 5.0, 6.0],
//!     [4.0, 5.0, 6.0],
//!     [1.0, 2.0, 3.0],
//! ]);
//! ```
//!
//! To select from anything after the 0th axis, you need a multi-dimensional
//! axis. See [GatherTo] and [SelectTo] docstrings for examples of this.
//!
//! But you can use [BroadcastTo] to make this easy! In this example we select
//! the same index from the 1st axis of a tensor:
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let r = t.select::<Rank1<2>, _>(dev.tensor(1).broadcast());
//! assert_eq!(r.array(), [2.0, 5.0]);
//! ```

mod binary;
mod compound;
mod nn;
mod reduction;
mod shape;
mod unary;

pub use binary::*;
pub use compound::*;
pub use nn::*;
pub use reduction::*;
pub use shape::*;
pub use unary::*;

mod backward;
mod device;
pub(crate) mod ops;

pub use backward::Backward;
pub use device::Device;

pub(crate) mod cpu_kernels;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_kernels;

// mod impl_mask;
// pub use impl_mask::*;
