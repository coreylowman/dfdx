//! The struct definitions for all TensorXD, [Tensor] trait, and more.
//!
//! At a high level a tensor consists of only three parts
//! 1. A [crate::unique_id::UniqueId] to track which gradients are associated with what tensors
//! 2. An Nd rust array stored in a [std::rc::Rc].
//! 3. A tape, which can either actually be a tape ([crate::gradients::OwnedTape]) or be empty ([crate::gradients::NoneTape]).
//!
//! # Creating tensors
//!
//! ### Use the tensor function
//!
//! See [tensor()].
//!
//! ```rust
//! # use dfdx::prelude::
//! let t = tensor([1.0, 2.0, 3.0]);
//! ```
//!
//! ### Use the TensorCreator trait
//!
//! See [TensorCreator].
//!
//! 1. With raw rust arrays use the [TensorCreator::new()] method.
//! ```rust
//! # use dfdx::prelude::*;
//! let t: Tensor1D<3> = TensorCreator::new([1.0, 2.0, 3.0]);
//! ```
//!
//! 2. Filled with 0s or 1s use [TensorCreator::zeros()] and [TensorCreator::ones()].
//! ```rust
//! # use dfdx::prelude::*;
//! let t: Tensor1D<5> = TensorCreator::zeros();
//! let q: Tensor2D<3, 2> = TensorCreator::ones();
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
//! # Accessing or modifying underlying data
//!
//! Use [HasArrayData::data()] and [HasArrayData::mut_data()] to view or modify the underlying arrays.
//!
//! ```rust
//! # use dfdx::prelude::*;
//! let mut t = Tensor1D::<3>::zeros();
//! assert_eq!(t.data(), &[0.0, 0.0, 0.0]);
//! t.mut_data()[1] = 2.0;
//! assert_eq!(t.data(), &[0.0, 2.0, 0.0]);
//! ```
//!
//! # Tracking gradients
//!
//! Use the [trace()] or [traced()] methods to add [crate::gradients::OwnedTape] to the [Tensor].
//! `.trace()` will clone the [crate::unique_id::UniqueId] & data, while `.traced()` will take ownership of
//! the tensor and return a version with an [crate::gradients::OwnedTape].
//!
//! Note that these two methods are only present for tensors without a tape already.
//!
//! ```rust
//! # use dfdx::prelude::*;
//! let t: Tensor1D<5, NoneTape> = Tensor1D::<5>::zeros();
//! let t_clone: Tensor1D<5, OwnedTape> = t.trace(); // copies t
//! let t: Tensor1D<5, OwnedTape> = t.traced(); // takes ownership of t
//! ```
//!
//! # Cloning/copying
//!
//! There are two primary methods for copying a tensor
//! 1. [Clone] is implemented for tensors without a tape. **NOTE** that the unique id is modified when a tensor is cloned
//! 2. [Tensor::duplicate()] is implemented for all tensors, it copies the [crate::unique_id::UniqueId], and returns a tensor with no tape.

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
mod into_tensor;
mod structs;

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
pub use into_tensor::*;
pub use structs::*;
