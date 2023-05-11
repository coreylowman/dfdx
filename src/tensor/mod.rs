//! The [Tensor] struct, [Cpu] & [Cuda] devices, and
//! traits like [ZerosTensor], [OnesTensor], [SampleTensor].
//!
//! At a high level a tensor is made up of:
//! 1. The [crate::shapes::Shape] of the array it stores
//! 2. The [crate::shapes::Dtype] of the elements of the array
//! 3. The [DeviceStorage] (e.g. [Cpu] or [Cuda]) that it uses to store the nd array
//! 4. A [Tape], which can either actually be a tape ([OwnedTape])
//!    or be empty ([NoneTape]).
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
//! let dev: Cpu = Cpu::seed_from_u64(0);
//! ```
//!
//! ```ignore
//! # use dfdx::prelude::*;
//! let dev: Cuda = Default::default();
//! let dev: Cuda = Cuda::seed_from_u64(1234);
//! let dev: Cuda = Cuda::try_build(0, 1234).unwrap();
//! ```
//!
//! # Creating tensors
//!
//! ### From arrays/vecs
//!
//! See [TensorFrom] & [TensorFromVec].
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let _ = dev.tensor([1.0, 2.0, 3.0]);
//! let _ = dev.tensor_from_vec(vec![1.0, 2.0, 3.0], (3, ));
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
//! # Tracing gradients
//!
//! Use the [Tensor::trace] or [Tensor::traced] methods to add [OwnedTape] to the [Tensor].
//! `.trace()` will clone the data, while `.traced()` will take ownership of
//! the tensor and return a version with an [OwnedTape].
//!
//! Note that these two methods are only present for tensors without a tape already.
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t: Tensor<Rank1<5>,f32, _> = dev.zeros();
//! let mut grads = t.alloc_grads();
//! let t_clone: Tensor<Rank1<5>, f32, _, OwnedTape<f32, Cpu>> = t.trace(grads);
//! ```
//!
//! ## Gradient Accumulation
//!
//! If you re-use the same gradients object without zero-ing out the gradients, you can
//! implement gradient accumulation!
//!
//! # Serialization using numpy
//!
//! See [Tensor::save_to_npy] and [Tensor::load_from_npy].
//!
//! You can also use [Tensor::write_to_npz] and [Tensor::read_from_npz] when working with
//! zip archives.
//!
//! # Allocation Caching
//!
//! By default, devices will cache allocations to reuse later. For example, the CPU will
//! cache `Vec<E>` allocations in an hash map, and try to re-use the allocation when
//! similarly sized data is used. This can result in significant speedups when memory
//! allocation patterns are repetitive. If this results in extra memory use due to
//! irregular allocation patterns there are two things you can do:
//!
//! 1. Call [DeviceStorage::empty_cache()], which will empty out all of the saved allocations.
//! 2. Disable the cache entirely by calling [DeviceStorage::disable_cache()]. This will
//! empty out any existing allocations and prevent any new ones from being cached.

pub(crate) mod cache;
pub(crate) mod cpu;
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
mod ghost;
mod gradients;
mod masks;
#[cfg(feature = "numpy")]
pub(crate) mod numpy;
#[cfg(feature = "safetensors")]
pub mod safetensors;
mod tensorlike;
mod unique_id;

pub(crate) mod storage_traits;
mod tensor_impls;

pub(crate) use ghost::GhostTensor;
pub(crate) use storage_traits::{OneFillStorage, ZeroFillStorage};
pub(crate) use tensorlike::Tensorlike;

pub use cpu::{Cpu, CpuError};
#[cfg(not(feature = "cuda"))]
pub type AutoDevice = Cpu;

#[cfg(feature = "cuda")]
pub(crate) use cuda::launch_cfg;
#[cfg(feature = "cuda")]
pub use cuda::{Cuda, CudaError};
#[cfg(feature = "cuda")]
pub type AutoDevice = Cuda;

pub use storage_traits::{AsArray, CopySlice, TensorFrom, TensorFromVec};
pub use storage_traits::{CacheSize, DeviceStorage, HasErr};
pub use storage_traits::{OnesTensor, SampleTensor, TriangleTensor, ZerosTensor};

pub use tensor_impls::{PutTape, SplitTape, Tensor, Trace, WithEmptyTape};
pub use tensor_impls::{Tensor0D, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D, Tensor6D};

pub(crate) use unique_id::unique_id;
pub use unique_id::UniqueId;

pub use gradients::{Gradients, Merge, NoneTape, OwnedTape, Tape};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::*;
    use crate::tests::*;
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

    #[test]
    fn test_upper_tri() {
        let dev: TestDevice = Default::default();
        let a: TestDtype = NumCast::from(42.0).unwrap();
        let z = TestDtype::zero();

        assert_eq!(dev.upper_tri::<Rank0>(a, None).array(), a);
        assert_eq!(dev.upper_tri::<Rank0>(a, 1).array(), z);
        assert_eq!(dev.upper_tri::<Rank1<3>>(a, None).array(), [a, a, a]);
        assert_eq!(dev.upper_tri::<Rank1<3>>(a, 1).array(), [z, a, a]);

        assert_eq!(
            dev.upper_tri::<Rank2<3, 4>>(a, None).array(),
            [[a, a, a, a], [z, a, a, a], [z, z, a, a]]
        );
        assert_eq!(
            dev.upper_tri::<Rank2<3, 1>>(a, None).array(),
            [[a], [z], [z]]
        );
        assert_eq!(dev.upper_tri::<Rank2<3, 1>>(a, 1).array(), [[z], [z], [z]]);
        assert_eq!(dev.upper_tri::<Rank2<3, 1>>(a, -1).array(), [[a], [a], [z]]);
        assert_eq!(
            dev.upper_tri::<Rank2<4, 4>>(a, -1).array(),
            [[a, a, a, a], [a, a, a, a], [z, a, a, a], [z, z, a, a]]
        );
        assert_eq!(
            dev.upper_tri::<Rank2<4, 4>>(a, -2).array(),
            [[a, a, a, a], [a, a, a, a], [a, a, a, a], [z, a, a, a]]
        );
        assert_eq!(
            dev.upper_tri::<Rank2<4, 3>>(a, 1).array(),
            [[z, a, a], [z, z, a], [z, z, z], [z, z, z]]
        );
        assert_eq!(
            dev.upper_tri::<Rank3<2, 5, 5>>(a, None).array(),
            [[
                [a, a, a, a, a],
                [z, a, a, a, a],
                [z, z, a, a, a],
                [z, z, z, a, a],
                [z, z, z, z, a]
            ]; 2]
        );
        assert_eq!(
            dev.upper_tri::<Rank3<4, 5, 5>>(a, 2).array(),
            [[
                [z, z, a, a, a],
                [z, z, z, a, a],
                [z, z, z, z, a],
                [z, z, z, z, z],
                [z, z, z, z, z]
            ]; 4]
        );
        assert_eq!(
            dev.upper_tri::<Rank4<3, 4, 5, 6>>(a, None).array(),
            [[[
                [a, a, a, a, a, a],
                [z, a, a, a, a, a],
                [z, z, a, a, a, a],
                [z, z, z, a, a, a],
                [z, z, z, z, a, a]
            ]; 4]; 3]
        );
    }

    #[test]
    fn test_lower_tri() {
        let dev: TestDevice = Default::default();
        let a: TestDtype = NumCast::from(42.0).unwrap();
        let z = TestDtype::zero();

        assert_eq!(dev.lower_tri::<Rank0>(a, None).array(), a);
        assert_eq!(dev.lower_tri::<Rank0>(a, -1).array(), z);
        assert_eq!(dev.lower_tri::<Rank1<3>>(a, None).array(), [a, z, z]);
        assert_eq!(dev.lower_tri::<Rank1<3>>(a, 1).array(), [a, a, z]);

        assert_eq!(
            dev.lower_tri::<Rank2<3, 4>>(a, None).array(),
            [[a, z, z, z], [a, a, z, z], [a, a, a, z]]
        );
        assert_eq!(
            dev.lower_tri::<Rank2<3, 1>>(a, None).array(),
            [[a], [a], [a]]
        );
        assert_eq!(dev.lower_tri::<Rank2<3, 1>>(a, 1).array(), [[a], [a], [a]]);
        assert_eq!(dev.lower_tri::<Rank2<3, 1>>(a, -1).array(), [[z], [a], [a]]);
        assert_eq!(
            dev.lower_tri::<Rank2<4, 4>>(a, -1).array(),
            [[z, z, z, z], [a, z, z, z], [a, a, z, z], [a, a, a, z]]
        );
        assert_eq!(
            dev.lower_tri::<Rank2<4, 4>>(a, -2).array(),
            [[z, z, z, z], [z, z, z, z], [a, z, z, z], [a, a, z, z]]
        );
        assert_eq!(
            dev.lower_tri::<Rank2<4, 3>>(a, 1).array(),
            [[a, a, z], [a, a, a], [a, a, a], [a, a, a]]
        );
        assert_eq!(
            dev.lower_tri::<Rank3<2, 5, 5>>(a, None).array(),
            [[
                [a, z, z, z, z],
                [a, a, z, z, z],
                [a, a, a, z, z],
                [a, a, a, a, z],
                [a, a, a, a, a]
            ]; 2]
        );
        assert_eq!(
            dev.lower_tri::<Rank3<4, 5, 5>>(a, 2).array(),
            [[
                [a, a, a, z, z],
                [a, a, a, a, z],
                [a, a, a, a, a],
                [a, a, a, a, a],
                [a, a, a, a, a]
            ]; 4]
        );
        assert_eq!(
            dev.lower_tri::<Rank4<3, 4, 5, 6>>(a, None).array(),
            [[[
                [a, z, z, z, z, z],
                [a, a, z, z, z, z],
                [a, a, a, z, z, z],
                [a, a, a, a, z, z],
                [a, a, a, a, a, z]
            ]; 4]; 3]
        );
    }
}
