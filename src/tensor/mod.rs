//! The struct definitions for all TensorXD, [Tensor] trait, and more.
//!
//! At a high level a tensor consists of only three parts
//! 1. A [UniqueId] to track which gradients are associated with what tensors
//! 2. An Nd rust array stored in a [Box].
//! 3. A tape, which can either actually be a tape ([OwnsTape]) or be empty ([NoTape]).
//!
//! Creating tensors:
//!
//! 1. With raw rust arrays use the [TensorCreator::new()] method.
//! ```rust
//! # use dfdx::prelude::*;
//! let t = Tensor1D::new([1.0, 2.0, 3.0]);
//! ```
//!
//! 2. Filled with 0s or 1s use [TensorCreator::zeros()] and [TensorCreator::ones()].
//! ```rust
//! # use dfdx::prelude::*;
//! let t: Tensor1D<5> = Tensor1D::zeros();
//! let q: Tensor2D<3, 2> = Tensor2D::ones();
//! ```
//!
//! 3. Filled with random data use [TensorCreator::rand()] and [TensorCreator::randn()].
//! ```rust
//! # use dfdx::prelude::*;
//! # use rand::prelude::*;
//! let mut rng = StdRng::seed_from_u64(0);
//! let a = Tensor1D::<3>::rand(&mut rng); // uniform random data
//! let b = Tensor2D::<4, 3>::randn(&mut rng); // gaussian random data
//! ```
//!
//! Accessing or modifying underlying data:
//!
//! Use [HasArrayData::data()] and [HasArrayData::mut_data()] to view or modify the underlying arrays.
//!
//! ```rust
//! # use dfdx::prelude::*;
//! let mut t = Tensor1D::<3>::zeros();
//! assert_eq!(t.data(), &[0.0, 0.0, 0.0]);
//! t.mut_data()[1] = 3.14;
//! assert_eq!(t.data(), &[0.0, 3.14, 0.0]);
//! ```
//!
//! Initiating gradient tracing:
//!
//! Use the `.trace()` or `.traced()` methods to add [OwnsTape] to the [Tensor].
//! `.trace()` will clone the [UniqueId] & data, while `.traced()` will take ownership of
//! the tensor and return a version with an [OwnsTape].
//!
//! Note that these two methods are only present for tensors without a tape already.
//!
//! ```rust
//! # use dfdx::prelude::*;
//! let t: Tensor1D<5, NoTape> = Tensor1D::<5>::zeros();
//! let t_clone: Tensor1D<5, OwnsTape> = t.trace(); // copies t
//! let t: Tensor1D<5, OwnsTape> = t.traced(); // takes ownership of t
//! ```
//!
//! Cloning/copying:
//!
//! There are two primary methods for copying a tensor
//! 1. [Clone] is implemented for tensors without a tape. **NOTE** that the unique id is modified when a tensor is cloned
//! 2. [Tensor::duplicate()] is implemented for all tensors, it copies the [UniqueId], and returns a tensor with no tape.

mod impl_backward;
mod impl_default;
mod impl_has_array;
mod impl_has_device;
mod impl_has_unique_id;
mod impl_phantom;
mod impl_put_tape;
mod impl_randomize;
mod impl_tensor;
mod impl_tensor_creator;
mod impl_trace;
mod impl_update_with_grads;
mod structs;

pub use impl_backward::*;
pub use impl_default::*;
pub use impl_has_array::*;
pub use impl_has_device::*;
pub use impl_has_unique_id::*;
pub use impl_phantom::*;
pub use impl_put_tape::*;
pub use impl_randomize::*;
pub use impl_tensor::*;
pub use impl_tensor_creator::*;
pub use impl_trace::*;
pub use impl_update_with_grads::*;
pub use structs::*;
