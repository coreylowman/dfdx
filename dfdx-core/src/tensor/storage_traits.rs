use rand::distributions::Distribution;
use rand_distr::{Standard, StandardNormal};
use std::vec::Vec;

use crate::shapes::*;

use super::{Error, Tensor};

pub trait RandomU64 {
    /// Generates a random u64 number
    fn random_u64(&self) -> u64;
}

/// Something that can store nd arrays for a given [Shape] and [Dtype]
pub trait Storage<E>: 'static + std::fmt::Debug + Default + Clone {
    /// Generic Storage type
    type Vec: 'static + std::fmt::Debug + Clone + Send + Sync;

    /// Allocates a gradient for the given nd array
    fn try_alloc_grad(&self, storage: &Self::Vec) -> Result<Self::Vec, Error> {
        self.try_alloc_len(self.len(storage))
    }

    fn try_alloc_len(&self, len: usize) -> Result<Self::Vec, Error>;

    fn tensor_to_vec<S: Shape, T>(&self, tensor: &Tensor<S, E, Self, T>) -> Vec<E>;

    fn len(&self, v: &Self::Vec) -> usize;
}

pub trait Synchronize {
    /// Blocks until all work on device to complete. Useful for benchmarking.
    fn synchronize(&self) {
        self.try_synchronize().unwrap()
    }

    /// Blocks until all work on device to complete. Useful for benchmarking.
    fn try_synchronize(&self) -> Result<(), Error>;
}

pub trait Cache {
    /// Enables the cache of the device.
    fn enable_cache(&self) {
        self.try_enable_cache().unwrap()
    }

    /// Tries to enable the cache of the device.
    fn try_enable_cache(&self) -> Result<(), Error>;

    /// Disables the cache of the device. This will also empty the cache
    /// if there are things in it. See [Cache::empty_cache] for
    /// more information.
    fn disable_cache(&self) {
        self.try_disable_cache().unwrap()
    }

    /// Tries to disable the cache of the device. See [Cache::disable_cache] for
    /// details of when this is useful.
    fn try_disable_cache(&self) -> Result<(), Error>;

    /// Empties the cache of the device.
    ///
    /// Currently devices will cache tensor allocations to avoid
    /// allocating and deallocating memory. This results is large
    /// speedups, but may potentially hold on to more memory than
    /// is actually being used.
    ///
    /// This method will empty the cache of the device, freeing
    /// all memory that is currently being held.
    fn empty_cache(&self) {
        self.try_empty_cache().unwrap();
    }

    /// Tries to empty the cache of the device. See [Cache::empty_cache] for
    /// details of when this is useful.
    fn try_empty_cache(&self) -> Result<(), Error>;
}

/// Internal trait - Represents something that can allocate its own gradient.
pub trait AllocGrad {
    type Gradient: 'static;
    fn try_alloc_grad(&self) -> Result<Self::Gradient, Error>;
}

impl<S: Shape, E, D: Storage<E>, T> AllocGrad for Tensor<S, E, D, T> {
    type Gradient = D::Vec;
    fn try_alloc_grad(&self) -> Result<Self::Gradient, Error> {
        self.device.try_alloc_grad(self.data.as_ref())
    }
}

/// Enables copying data into and out of tensors
pub trait CopySlice<E>: Storage<E> {
    fn copy_from<S: Shape, T>(dst: &mut Tensor<S, E, Self, T>, src: &[E]);
    fn copy_into<S: Shape, T>(src: &Tensor<S, E, Self, T>, dst: &mut [E]);
}

impl<S: Shape, E, D: CopySlice<E>, T> Tensor<S, E, D, T> {
    /// Copy *physical* data from a slice - **panics** if there are not enough elements in the slice.
    ///
    /// ```rust
    /// # use dfdx_core::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let data = [1.0, 2.0, 3.0, 4.0];
    /// let mut t: Tensor<Rank2<2, 2>, f32, _> = dev.zeros();
    /// t.copy_from(&data);
    /// assert_eq!(t.array(), [[1.0, 2.0], [3.0, 4.0]]);
    /// ```
    pub fn copy_from(&mut self, src: &[E]) {
        D::copy_from(self, src);
    }

    /// Copy *physical* data into a slice - **panics** if there are not enough elements in the tensor.
    ///
    /// ```rust
    /// # use dfdx_core::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let t: Tensor<Rank2<2, 2>, f32, _> = dev.tensor([[1.0, 2.0], [3.0, 4.0]]);
    /// let mut data = [0.0; 4];
    /// t.copy_into(&mut data);
    /// assert_eq!(data, [1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn copy_into(&self, dst: &mut [E]) {
        D::copy_into(self, dst);
    }
}

/// Construct tensors filled with zeros.
pub trait ZerosTensor<E>: Storage<E> {
    /// Creates a tensor filled with zeros.
    /// ```rust
    /// # use dfdx_core::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
    /// ```
    fn zeros<S: ConstShape>(&self) -> Tensor<S, E, Self> {
        self.try_zeros_like::<S>(&Default::default()).unwrap()
    }

    /// Fallible version of [ZerosTensor::zeros]
    fn try_zeros<S: ConstShape>(&self) -> Result<Tensor<S, E, Self>, Error> {
        self.try_zeros_like::<S>(&Default::default())
    }

    /// Build the tensor with a shape given by something else.
    ///
    /// Given a shape directly:
    /// ```rust
    /// # use dfdx_core::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<(usize, Const<3>), f32, _> = dev.zeros_like(&(5, Const));
    /// ```
    ///
    /// Given another tensor:
    /// ```rust
    /// # use dfdx_core::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
    /// let b: Tensor<Rank2<2, 3>, f32, _> = dev.zeros_like(&a);
    /// ```
    fn zeros_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self> {
        self.try_zeros_like(src).unwrap()
    }

    /// Fallible version of [ZerosTensor::zeros_like]
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Error>;
}

pub trait ZeroFillStorage<E>: Storage<E> {
    fn try_fill_with_zeros(&self, storage: &mut Self::Vec) -> Result<(), Error>;
}

/// View or mutate a [Storage::Vec] object.
pub trait WithStorage<E>: Storage<E> {
    /// View the values by each element.
    fn try_element_view<F: FnMut(&E)>(&self, storage: &Self::Vec, f: F) -> Result<(), Error>;
    /// View the values by a [Vec].
    fn try_view<F: FnMut(&[E])>(&self, storage: &Self::Vec, f: F) -> Result<(), Error>;
    /// Mutates the values by each element.
    fn try_element_map<F: FnMut(E) -> E>(&self, storage: &mut Self::Vec, f: F)
        -> Result<(), Error>;
    /// Mutates a clone of the values.
    ///
    /// If `Some` is returned, replaces the changed values back into the object.  
    /// Otherwise if `None` is returned, the changed values are discarded and the object stays intact.
    fn try_map<F: FnMut(Vec<E>) -> Option<Vec<E>>>(
        &self,
        storage: &mut Self::Vec,
        f: F,
    ) -> Result<(), Error>;
}

/// Construct tensors filled with ones.
pub trait OnesTensor<E>: Storage<E> {
    /// Creates a tensor filled with ones.
    /// ```rust
    /// # use dfdx_core::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32, _> = dev.ones();
    /// ```
    fn ones<S: ConstShape>(&self) -> Tensor<S, E, Self> {
        self.try_ones_like::<S>(&Default::default()).unwrap()
    }

    /// Fallible version of [OnesTensor::ones]
    fn try_ones<S: ConstShape>(&self) -> Result<Tensor<S, E, Self>, Error> {
        self.try_ones_like::<S>(&Default::default())
    }

    /// Build the tensor with a shape given by something else.
    ///
    /// Given a shape directly:
    /// ```rust
    /// # use dfdx_core::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<(usize, Const<3>), f32, _> = dev.ones_like(&(5, Const));
    /// ```
    ///
    /// Given another tensor:
    /// ```rust
    /// # use dfdx_core::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32, _> = dev.ones();
    /// let b: Tensor<_, f32, _> = dev.ones_like(&a);
    /// ```
    fn ones_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self> {
        self.try_ones_like(src).unwrap()
    }

    /// Fallible version of [OnesTensor::ones_like]
    fn try_ones_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Error>;
}

pub trait OneFillStorage<E>: Storage<E> {
    fn try_fill_with_ones(&self, storage: &mut Self::Vec) -> Result<(), Error>;
}

/// Build upper & lower triangle tensors.
pub trait TriangleTensor<E>: Storage<E> {
    /// Build a tensor containing the upper triangle part of each lowest 2D matrix
    /// set to the given value, along the given diagonal. The other values will be `E::default()`.
    ///
    /// Given a 2D matrix `M x N`, diagonal values will shift the values in the
    /// `-M/+N` direction.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// # use dfdx_core::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<3, 3>, f32, _> = dev.upper_tri(1.0, None);
    /// assert_eq!(a.array(),
    ///     [[1.0, 1.0, 1.0],
    ///      [0.0, 1.0, 1.0],
    ///      [0.0, 0.0, 1.0]]
    /// );
    /// let b: Tensor<_, f32, _> = dev.upper_tri_like(&a, 1.0, -1);
    /// assert_eq!(b.array(),
    ///     [[1.0, 1.0, 1.0],
    ///      [1.0, 1.0, 1.0],
    ///      [0.0, 1.0, 1.0]]
    /// );
    /// let c: Tensor<_, f32, _> = dev.upper_tri_like(&b, 1.0, 1);
    /// assert_eq!(c.array(),
    ///     [[0.0, 1.0, 1.0],
    ///      [0.0, 0.0, 1.0],
    ///      [0.0, 0.0, 0.0]]
    /// );
    /// ```
    fn upper_tri<S: ConstShape>(
        &self,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Tensor<S, E, Self> {
        self.try_upper_tri_like::<S>(&Default::default(), val, diagonal)
            .unwrap()
    }

    /// Fallible version of [TriangleTensor::upper_tri]
    fn try_upper_tri<S: ConstShape>(
        &self,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        self.try_upper_tri_like::<S>(&Default::default(), val, diagonal)
    }

    /// Build an upper triangular tensor with the given shape. See [TriangleTensor::upper_tri].
    fn upper_tri_like<S: HasShape>(
        &self,
        src: &S,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Tensor<S::Shape, E, Self> {
        self.try_upper_tri_like(src, val, diagonal).unwrap()
    }

    /// Fallible version of [TriangleTensor::upper_tri_like]
    fn try_upper_tri_like<S: HasShape>(
        &self,
        src: &S,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Tensor<S::Shape, E, Self>, Error>;

    /// Build a tensor containing the lower triangle part of each lowest 2D matrix
    /// set to the given value, along the given diagonal. The other values will be `E::default()`.
    ///
    /// Given a 2D matrix `M x N`, diagonal values will shift the values in the
    /// `-M/+N` direction.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// # use dfdx_core::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<3, 3>, f32, _> = dev.lower_tri(1.0, None);
    /// assert_eq!(a.array(),
    ///     [[1.0, 0.0, 0.0],
    ///      [1.0, 1.0, 0.0],
    ///      [1.0, 1.0, 1.0]]
    /// );
    /// let b: Tensor<_, f32, _> = dev.lower_tri_like(&a, 1.0, -1);
    /// assert_eq!(b.array(),
    ///     [[0.0, 0.0, 0.0],
    ///      [1.0, 0.0, 0.0],
    ///      [1.0, 1.0, 0.0]]
    /// );
    /// let c: Tensor<_, f32, _> = dev.lower_tri_like(&b, 1.0, 1);
    /// assert_eq!(c.array(),
    ///     [[1.0, 1.0, 0.0],
    ///      [1.0, 1.0, 1.0],
    ///      [1.0, 1.0, 1.0]]
    /// );
    /// ```
    fn lower_tri<S: ConstShape>(
        &self,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Tensor<S, E, Self> {
        self.try_lower_tri_like::<S>(&Default::default(), val, diagonal)
            .unwrap()
    }

    /// Fallible version of [TriangleTensor::lower_tri]
    fn try_lower_tri<S: ConstShape>(
        &self,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        self.try_lower_tri_like::<S>(&Default::default(), val, diagonal)
    }

    /// Build a lower triangular tensor with the given shape. See [TriangleTensor::lower_tri].
    fn lower_tri_like<S: HasShape>(
        &self,
        src: &S,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Tensor<S::Shape, E, Self> {
        self.try_lower_tri_like(src, val, diagonal).unwrap()
    }

    /// Fallible version of [TriangleTensor::lower_tri_like]
    fn try_lower_tri_like<S: HasShape>(
        &self,
        src: &S,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Tensor<S::Shape, E, Self>, Error>;
}

/// Constructs tensors filled with random values from a given distribution.
pub trait SampleTensor<E>: Storage<E> {
    /// Samples a const tensor from a uniform distribution
    fn sample_uniform<S: ConstShape>(&self) -> Tensor<S, E, Self>
    where
        Standard: Distribution<E>,
    {
        self.sample::<S, _>(Standard)
    }
    /// Samples a tensor with a given shape from a uniform distribution
    fn sample_uniform_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self>
    where
        Standard: Distribution<E>,
    {
        self.sample_like::<S, _>(src, Standard)
    }

    /// Samples a const tensor from a normal distribution
    fn sample_normal<S: ConstShape>(&self) -> Tensor<S, E, Self>
    where
        StandardNormal: Distribution<E>,
    {
        self.sample::<S, _>(StandardNormal)
    }
    /// Samples a tensor with a given shape from a normal distribution
    fn sample_normal_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self>
    where
        StandardNormal: Distribution<E>,
    {
        self.sample_like::<S, _>(src, StandardNormal)
    }

    /// Samples a const tensor from a given distribution.
    fn sample<S: ConstShape, D: Distribution<E>>(&self, distr: D) -> Tensor<S, E, Self> {
        self.try_sample_like::<S, D>(&Default::default(), distr)
            .unwrap()
    }
    /// Fallibly samples a const tensor from a given distribution.
    fn try_sample<S: ConstShape, D: Distribution<E>>(
        &self,
        distr: D,
    ) -> Result<Tensor<S, E, Self>, Error> {
        self.try_sample_like::<S, D>(&Default::default(), distr)
    }

    /// Samples a tensor with a given shape from a given distribution.
    fn sample_like<S: HasShape, D: Distribution<E>>(
        &self,
        src: &S,
        distr: D,
    ) -> Tensor<S::Shape, E, Self> {
        self.try_sample_like(src, distr).unwrap()
    }
    /// Fallibly samples a tensor with a given shape from a given distribution.
    fn try_sample_like<S: HasShape, D: Distribution<E>>(
        &self,
        src: &S,
        distr: D,
    ) -> Result<Tensor<S::Shape, E, Self>, Error>;

    /// Fills tensor `Storage<E>` with data from a given distribution
    fn try_fill_with_distr<D: Distribution<E>>(
        &self,
        storage: &mut Self::Vec,
        distr: D,
    ) -> Result<(), Error>;
}

pub trait TensorToArray<S: Shape, E>: Storage<E> {
    type Array: std::fmt::Debug + PartialEq;
    fn tensor_to_array<T>(&self, tensor: &Tensor<S, E, Self, T>) -> Self::Array;
}

pub trait AsArray {
    type Array: std::fmt::Debug + PartialEq;
    fn array(&self) -> Self::Array;
}

impl<S: Shape, E, D: TensorToArray<S, E>, T> AsArray for Tensor<S, E, D, T> {
    type Array = D::Array;
    /// Convert tensors to rust arrays
    fn array(&self) -> Self::Array {
        self.device.tensor_to_array(self)
    }
}

impl<S: Shape, E, D: Storage<E>, T> Tensor<S, E, D, T> {
    pub fn as_vec(&self) -> std::vec::Vec<E> {
        self.device.tensor_to_vec(self)
    }
}

/// Construct tensors from rust vectors. This trait is only used to implement TensorFrom.
pub trait TensorFromVec<E>: Storage<E> {
    fn tensor_from_vec<S: Shape>(&self, src: Vec<E>, shape: S) -> Tensor<S, E, Self> {
        self.try_tensor_from_vec::<S>(src, shape).unwrap()
    }

    fn try_tensor_from_vec<S: Shape>(
        &self,
        src: Vec<E>,
        shape: S,
    ) -> Result<Tensor<S, E, Self>, Error>;
}

impl<S: Shape, E, D: Storage<E>, T> Tensor<S, E, D, T> {
    /// Clones the tensor onto a different device.
    pub fn to_device<Dst: TensorFromVec<E>>(&self, device: &Dst) -> Tensor<S, E, Dst> {
        self.try_to_device(device).unwrap()
    }

    /// Fallibly clones the tensor onto a different device.
    pub fn try_to_device<Dst: TensorFromVec<E>>(
        &self,
        device: &Dst,
    ) -> Result<Tensor<S, E, Dst>, Error> {
        let buf = self.as_vec();
        device.try_tensor_from_vec(buf, self.shape)
    }
}

/// Construct tensors from rust data
pub trait TensorFrom<Src, S: Shape, E>: Storage<E> {
    /// Create a tensor from rust data
    /// ```rust
    /// # use dfdx_core::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let _: Tensor<Rank2<2, 3>, f32, Cpu> = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let _: Tensor<Rank2<2, 3>, f32, Cpu> = dev.tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// // Note: arguments are in a tuple, and this syntax should only be used when creating
    /// // tensors with a dynamic shape
    /// let _ = dev.tensor((vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]));
    /// ```
    fn tensor(&self, src: Src) -> Tensor<S, E, Self> {
        self.try_tensor(src).unwrap()
    }
    /// Fallible version of [TensorFrom::tensor]
    fn try_tensor(&self, src: Src) -> Result<Tensor<S, E, Self>, Error>;
}

impl<E, D: TensorFromVec<E>> TensorFrom<E, Rank0, E> for D {
    fn try_tensor(&self, src: E) -> Result<Tensor<Rank0, E, Self>, Error> {
        self.try_tensor_from_vec(vec![src], ())
    }
}

impl<E: Copy, const M: usize, D: TensorFromVec<E>> TensorFrom<[E; M], Rank1<M>, E> for D {
    fn try_tensor(&self, src: [E; M]) -> Result<Tensor<Rank1<M>, E, Self>, Error> {
        self.try_tensor(&src)
    }
}

impl<E: Copy, const M: usize, D: TensorFromVec<E>> TensorFrom<&[E; M], Rank1<M>, E> for D {
    fn try_tensor(&self, src: &[E; M]) -> Result<Tensor<Rank1<M>, E, Self>, Error> {
        self.try_tensor_from_vec(src.to_vec(), (Const::<M>,))
    }
}

impl<E: Copy, const M: usize, const N: usize, D: TensorFromVec<E>>
    TensorFrom<[[E; N]; M], Rank2<M, N>, E> for D
{
    fn try_tensor(&self, src: [[E; N]; M]) -> Result<Tensor<Rank2<M, N>, E, Self>, Error> {
        let vec: Vec<E> = src.iter().flat_map(|v| v.iter().copied()).collect();

        self.try_tensor_from_vec(vec, (Const::<M>, Const::<N>))
    }
}

impl<E: Copy, const M: usize, const N: usize, const O: usize, D: TensorFromVec<E>>
    TensorFrom<[[[E; O]; N]; M], Rank3<M, N, O>, E> for D
{
    fn try_tensor(&self, src: [[[E; O]; N]; M]) -> Result<Tensor<Rank3<M, N, O>, E, Self>, Error> {
        let vec: Vec<E> = src
            .iter()
            .flat_map(|v| v.iter())
            .flat_map(|v| v.iter().copied())
            .collect();

        self.try_tensor_from_vec(vec, (Const::<M>, Const::<N>, Const::<O>))
    }
}

impl<
        E: Copy,
        const M: usize,
        const N: usize,
        const O: usize,
        const P: usize,
        D: TensorFromVec<E>,
    > TensorFrom<[[[[E; P]; O]; N]; M], Rank4<M, N, O, P>, E> for D
{
    fn try_tensor(
        &self,
        src: [[[[E; P]; O]; N]; M],
    ) -> Result<Tensor<Rank4<M, N, O, P>, E, Self>, Error> {
        let vec: Vec<E> = src
            .iter()
            .flat_map(|v| v.iter())
            .flat_map(|v| v.iter())
            .flat_map(|v| v.iter().copied())
            .collect();

        self.try_tensor_from_vec(vec, (Const::<M>, Const::<N>, Const::<O>, Const::<P>))
    }
}

impl<E, S: ConstShape, D: TensorFromVec<E>> TensorFrom<Vec<E>, S, E> for D {
    fn try_tensor(&self, src: Vec<E>) -> Result<Tensor<S, E, Self>, Error> {
        self.try_tensor_from_vec(src, S::default())
    }
}

impl<E, S: Shape, D: TensorFromVec<E>> TensorFrom<(Vec<E>, S), S, E> for D {
    fn try_tensor(&self, (src, shape): (Vec<E>, S)) -> Result<Tensor<S, E, Self>, Error> {
        self.try_tensor_from_vec(src, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{prelude::SumTo, tensor::*, tensor_ops::Backward, tests::*};
    use core::ops::Mul;

    #[test]
    fn test_map_grads() {
        let dev: TestDevice = Default::default();
        let x1 = dev.tensor([1., 1., 1., 1., 1., 1.]).to_dtype::<TestDtype>();
        let x2 = dev
            .tensor([-3., -2., -1., 1., 2., 3.])
            .to_dtype::<TestDtype>();
        let loss = x1.leaky_trace().mul(x2).try_sum().unwrap();
        let mut grads = loss.backward();
        let grads_x1 = grads.get_mut(&x1);

        let mut acc = 0.;
        let map_element = |e| {
            acc += 1.;
            e + acc
        };
        let map_vec = |v: Vec<_>| Some(v.into_iter().map(|e| e * 0.5).collect());

        let (g1, g2, g3);
        let r1 = vec![-3., -2., -1., 1., 2., 3.];
        let r2 = vec![-2., 0., 2., 5., 7., 9.];
        let r3 = vec![-1., 0., 1., 2.5, 3.5, 4.5];

        #[cfg(feature = "cuda")]
        {
            g1 = dev.dev.dtoh_sync_copy(grads_x1).unwrap();
            dev.try_element_map(grads_x1, map_element).unwrap();
            g2 = dev.dev.dtoh_sync_copy(grads_x1).unwrap();
            dev.try_map(grads_x1, map_vec).unwrap();
            g3 = dev.dev.dtoh_sync_copy(grads_x1).unwrap();
        };
        #[cfg(feature = "webgpu")]
        {
            g1 = todo!();
            dev.try_element_map(grads_x1, map_element).unwrap();
            g2 = todo!();
            dev.try_map(grads_x1, map_vec).unwrap();
            g3 = todo!();
        };
        #[cfg(not(any(feature = "cuda", feature = "webgpu")))]
        {
            g1 = grads_x1.data.clone();
            dev.try_element_map(grads_x1, map_element).unwrap();
            g2 = grads_x1.data.clone();
            dev.try_map(grads_x1, map_vec).unwrap();
            g3 = grads_x1.data.clone();
        };
        assert_eq!(g1, r1);
        assert_eq!(g2, r2);
        assert_eq!(g3, r3);
    }
}
