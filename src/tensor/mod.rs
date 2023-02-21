//! The [Tensor] struct, [Cpu] device, and
//! traits like [ZerosTensor], [OnesTensor], [SampleTensor].
//!
//! At a high level a tensor is made up of:
//! 1. The [crate::shapes::Shape] of the array it stores
//! 2. The [crate::shapes::Dtype] of the elements of the array
//! 3. The [DeviceStorage] (e.g. [Cpu]) that it uses to store the nd array
//! 4. A [crate::gradients::Tape], which can either actually be a tape ([crate::gradients::OwnedTape])
//!    or be empty ([crate::gradients::NoneTape]).
//!
//! Which are all generic parameters of [Tensor]. See the type's docstring for more info
//!
//! # Creating a device
//!
//! In order to do anything with tensors, you first need to construct the device that they will be stored on:
//!
//! ```rust
//! # use dfdx::prelude::*;
//! let dev: Cpu = Default::default();
//! ```
//!
//! # Creating tensors
//!
//! ### From arrays
//!
//! See [TensorFrom].
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t = dev.tensor([1.0, 2.0, 3.0]);
//! ```
//!
//! ### Filled with 0s or 1s
//!
//! See [ZerosTensor] and [OnesTensor].
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let _: Tensor<Rank1<5>,f32 , _> = dev.zeros();
//! let _: Tensor<Rank2<3, 2>, f32, _> = dev.ones();
//! ```
//!
//! ### Filled with random data
//!
//! See [SampleTensor]
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let _: Tensor<Rank1<5>, f32, _> = dev.sample_uniform();
//! let _: Tensor<Rank2<3, 5>, f32, _> = dev.sample_normal();
//! // or pass in actual distributions
//! let _: Tensor<Rank1<3>, f32, _> = dev.sample(rand_distr::Standard);
//! let _: Tensor<Rank2<4, 3>, f32, _> = dev.sample(rand_distr::StandardNormal);
//! ```
//!
//! ### Copy data from slices
//!
//! You can use [Tensor::copy_from] and [Tensor::copy_into] to copy data into a tensor:
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let mut a: Tensor<Rank1<1000>, f32, _> = dev.zeros();
//! let buf: Vec<f32> = vec![1.0; 1000];
//! a.copy_from(&buf);
//! ```
//!
//! # Modifying an already constructed tensor
//!
//! There are only a few ways to do this, as normally you should just create a new tensor with tensor_ops.
//!
//! See [Tensor::fill_with_zeros], [Tensor::fill_with_ones], [Tensor::fill_with_distr]
//!
//! # Converting tensors to rust arrays
//!
//! Since the way tensors are stored is opaque to users (driven by whatever device the tensor is stored on),
//! use the [AsArray] trait to convert tensors to actual rust arrays if you want to work
//! with them directly.
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
//! let t: [[f32; 3]; 2] = t.array();
//! ```
//!
//! # Tracking gradients
//!
//! Use the [Tensor::trace] or [Tensor::traced] methods to add [crate::gradients::OwnedTape] to the [Tensor].
//! `.trace()` will clone the [crate::unique_id::UniqueId] & data, while `.traced()` will take ownership of
//! the tensor and return a version with an [crate::gradients::OwnedTape].
//!
//! Note that these two methods are only present for tensors without a tape already.
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t: Tensor<Rank1<5>,f32, _> = dev.zeros();
//! let t_clone: Tensor<Rank1<5>, _, _, OwnedTape<_>> = t.trace(); // copies t
//! let t = t.traced(); // takes ownership of t
//! ```
//!
//! # Serialization using numpy
//!
//! See [Tensor::save_to_npy] and [Tensor::load_from_npy].
//!
//! You can also use [Tensor::write_to_npz] and [Tensor::read_from_npz] when working with
//! zip archives.

pub(crate) mod cpu;
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "numpy")]
pub(crate) mod numpy;
pub(crate) mod storage_traits;
mod tensor_impls;
pub(crate) mod visitors; // TODO pub?

pub(crate) use storage_traits::{OneFillStorage, ZeroFillStorage};

pub use cpu::{Cpu, CpuError};

#[cfg(feature = "cuda")]
pub use cuda::{Cuda, CudaError};

pub use storage_traits::{AsArray, AsVec, CopySlice, TensorFrom};
pub use storage_traits::{DeviceStorage, HasErr};
pub use storage_traits::{OnesTensor, SampleTensor, ZerosTensor};

#[cfg(feature = "cuda")]
pub use tensor_impls::OnCuda;
pub use tensor_impls::{OnCpu, OnDevice, PutTape, SplitTape, Tensor, ToDevice};
pub use tensor_impls::{Tensor0D, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D, Tensor6D};

pub use visitors::{NumParams, ResetParams};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::*;
    use crate::tests::TestDevice;
    use crate::unique_id::{unique_id, UniqueId};
    use std::collections::HashSet;

    #[test]
    fn test_id() {
        let dev: TestDevice = Default::default();

        let mut ids: HashSet<UniqueId> = Default::default();
        ids.insert(unique_id());

        let x: Tensor<Rank0, f32, _> = dev.zeros();
        assert!(!ids.contains(&x.id));
        ids.insert(x.id);

        let x: Tensor<Rank0, f32, _> = dev.zeros();
        assert!(!ids.contains(&x.id));
        ids.insert(x.id);

        let x: Tensor<Rank1<5>, f32, _> = dev.zeros();
        assert!(!ids.contains(&x.id));
        ids.insert(x.id);

        let x: Tensor<Rank2<3, 2>, f32, _> = dev.ones();
        assert!(!ids.contains(&x.id));
        ids.insert(x.id);

        let x: Tensor<Rank3<4, 3, 2>, f32, _> = dev.sample(rand_distr::Standard);
        assert!(!ids.contains(&x.id));
        ids.insert(x.id);
    }

    #[test]
    fn test_ids_with_clone() {
        let dev: TestDevice = Default::default();
        let t1: Tensor<Rank1<32>, f32, _> = dev.zeros();
        let t2 = t1.clone();
        assert_eq!(t1.id, t2.id);
    }

    #[test]
    fn test_ids_with_split_and_put() {
        let dev: TestDevice = Default::default();
        let t1: Tensor<Rank1<32>, f32, _> = dev.zeros();
        let t1_id = t1.id;
        let (t2, tape) = t1.split_tape();
        assert_eq!(t2.id, t1_id);
        let t3 = t2.put_tape(tape);
        assert_eq!(t3.id, t1_id);
    }

    #[test]
    fn test_zeros() {
        let dev: TestDevice = Default::default();
        let x: Tensor<Rank2<3, 2>, f32, _> = dev.zeros();
        assert_eq!(x.array(), [[0.0; 2]; 3]);
    }

    #[test]
    fn test_ones() {
        let dev: TestDevice = Default::default();
        let x: Tensor<Rank2<3, 2>, f32, _> = dev.ones();
        assert_eq!(x.array(), [[1.0; 2]; 3]);
    }

    #[test]
    fn test_convert_array() {
        let dev: TestDevice = Default::default();
        let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let t = dev.tensor(a);
        assert_eq!(t.array(), a);
    }

    #[test]
    fn test_convert_slice() {
        let dev: TestDevice = Default::default();
        let data = [1.0, 2.0, 3.0, 4.0];
        let mut t: Tensor<Rank2<2, 2>, f32, _> = dev.zeros();
        t.copy_from(&data);
        assert_eq!(t.array(), [[1.0, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn fuzz_test_rand() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank1<1000>, f32, _> = dev.sample_uniform();
        for v in t.as_vec() {
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_sample_normal() {
        let dev: TestDevice = Default::default();
        let _: Tensor<Rank1<1000>, f32, _> = dev.sample_normal();
    }
}
