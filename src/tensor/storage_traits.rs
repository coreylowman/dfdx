use rand::distributions::Distribution;
use rand_distr::{Standard, StandardNormal};
use std::vec::Vec;

use crate::shapes::*;

use super::Tensor;

/// Represents something that has an error associated type
pub trait HasErr: Sized {
    type Err: std::fmt::Debug + std::fmt::Display;
}

/// Convert tensors to [std::vec::Vec]
pub trait AsVec<E> {
    fn as_vec(&self) -> std::vec::Vec<E>;
}

/// Something that can store nd arrays for a given [Shape] and [Dtype]
pub trait DeviceStorage: 'static + std::fmt::Debug + Default + Clone + HasErr {
    /// Generic storage type
    type Vec<E: Unit>: 'static + std::fmt::Debug + Clone + Send + Sync;

    /// Generates a random u64 number
    fn random_u64(&self) -> u64;

    /// Allocates a gradient for the given nd array
    fn try_alloc_grad<E: Unit>(&self, storage: &Self::Vec<E>) -> Result<Self::Vec<E>, Self::Err> {
        self.try_alloc_len(self.len(storage))
    }

    fn try_alloc_len<E: Unit>(&self, len: usize) -> Result<Self::Vec<E>, Self::Err>;

    fn tensor_to_vec<S: Shape, E: Unit, T>(&self, tensor: &Tensor<S, E, Self, T>) -> Vec<E>;

    fn len<E: Unit>(&self, v: &Self::Vec<E>) -> usize;

    /// Blocks until all work on device to complete. Useful for benchmarking.
    fn synchronize(&self) {
        self.try_synchronize().unwrap()
    }

    /// Blocks until all work on device to complete. Useful for benchmarking.
    fn try_synchronize(&self) -> Result<(), Self::Err>;

    /// Enables the cache of the device.
    fn enable_cache(&self) {
        self.try_enable_cache().unwrap()
    }

    /// Tries to enable the cache of the device.
    fn try_enable_cache(&self) -> Result<(), Self::Err>;

    /// Disables the cache of the device. This will also empty the cache
    /// if there are things in it. See [DeviceStorage::empty_cache] for
    /// more information.
    fn disable_cache(&self) {
        self.try_disable_cache().unwrap()
    }

    /// Tries to disable the cache of the device. See [DeviceStorage::disable_cache] for
    /// details of when this is useful.
    fn try_disable_cache(&self) -> Result<(), Self::Err>;

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

    /// Tries to empty the cache of the device. See [DeviceStorage::empty_cache] for
    /// details of when this is useful.
    fn try_empty_cache(&self) -> Result<(), Self::Err>;
}

/// Internal trait - Represents something that can allocate its own gradient.
pub trait AllocGrad: HasErr {
    type Gradient: 'static;
    fn try_alloc_grad(&self) -> Result<Self::Gradient, Self::Err>;
}

impl<S: Shape, E: Unit, D: DeviceStorage, T> AllocGrad for Tensor<S, E, D, T> {
    type Gradient = D::Vec<E>;
    fn try_alloc_grad(&self) -> Result<Self::Gradient, D::Err> {
        self.device.try_alloc_grad(self.data.as_ref())
    }
}

/// Enables copying data into and out of tensors
pub trait CopySlice<E: Unit>: DeviceStorage {
    fn copy_from<S: Shape, T>(dst: &mut Tensor<S, E, Self, T>, src: &[E]);
    fn copy_into<S: Shape, T>(src: &Tensor<S, E, Self, T>, dst: &mut [E]);
}

impl<S: Shape, E: Unit, D: CopySlice<E>, T> Tensor<S, E, D, T> {
    /// Copy *physical* data from a slice - **panics** if there are not enough elements in the slice.
    ///
    /// ```rust
    /// # use dfdx::prelude::*;
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
    /// # use dfdx::prelude::*;
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
pub trait ZerosTensor<E: Unit>: DeviceStorage {
    /// Creates a tensor filled with zeros.
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
    /// ```
    fn zeros<S: ConstShape>(&self) -> Tensor<S, E, Self> {
        self.try_zeros_like::<S>(&Default::default()).unwrap()
    }

    /// Fallible version of [ZerosTensor::zeros]
    fn try_zeros<S: ConstShape>(&self) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_zeros_like::<S>(&Default::default())
    }

    /// Build the tensor with a shape given by something else.
    ///
    /// Given a shape directly:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<(usize, Const<3>), f32, _> = dev.zeros_like(&(5, Const));
    /// ```
    ///
    /// Given another tensor:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
    /// let b: Tensor<Rank2<2, 3>, f32, _> = dev.zeros_like(&a);
    /// ```
    fn zeros_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self> {
        self.try_zeros_like(src).unwrap()
    }

    /// Fallible version of [ZerosTensor::zeros_like]
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;
}

pub trait ZeroFillStorage<E: Unit>: DeviceStorage {
    fn try_fill_with_zeros(&self, storage: &mut Self::Vec<E>) -> Result<(), Self::Err>;
}

/// Construct tensors filled with ones.
pub trait OnesTensor<E: Unit>: DeviceStorage {
    /// Creates a tensor filled with ones.
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32, _> = dev.ones();
    /// ```
    fn ones<S: ConstShape>(&self) -> Tensor<S, E, Self> {
        self.try_ones_like::<S>(&Default::default()).unwrap()
    }

    /// Fallible version of [OnesTensor::ones]
    fn try_ones<S: ConstShape>(&self) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_ones_like::<S>(&Default::default())
    }

    /// Build the tensor with a shape given by something else.
    ///
    /// Given a shape directly:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<(usize, Const<3>), f32, _> = dev.ones_like(&(5, Const));
    /// ```
    ///
    /// Given another tensor:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32, _> = dev.ones();
    /// let b: Tensor<_, f32, _> = dev.ones_like(&a);
    /// ```
    fn ones_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self> {
        self.try_ones_like(src).unwrap()
    }

    /// Fallible version of [OnesTensor::ones_like]
    fn try_ones_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;
}

pub trait OneFillStorage<E: Unit>: DeviceStorage {
    fn try_fill_with_ones(&self, storage: &mut Self::Vec<E>) -> Result<(), Self::Err>;
}

/// Build upper & lower triangle tensors.
pub trait TriangleTensor<E: Unit>: DeviceStorage {
    /// Build a tensor containing the upper triangle part of each lowest 2D matrix
    /// set to the given value, along the given diagonal. The other values will be `E::default()`.
    ///
    /// Given a 2D matrix `M x N`, diagonal values will shift the values in the
    /// `-M/+N` direction.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// # use dfdx::prelude::*;
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
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
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
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;

    /// Build a tensor containing the lower triangle part of each lowest 2D matrix
    /// set to the given value, along the given diagonal. The other values will be `E::default()`.
    ///
    /// Given a 2D matrix `M x N`, diagonal values will shift the values in the
    /// `-M/+N` direction.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// # use dfdx::prelude::*;
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
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
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
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;
}

/// Constructs tensors filled with random values from a given distribution.
pub trait SampleTensor<E: Unit>: DeviceStorage {
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
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
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
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;

    /// Fills tensor storage with data from a given distribution
    fn try_fill_with_distr<D: Distribution<E>>(
        &self,
        storage: &mut Self::Vec<E>,
        distr: D,
    ) -> Result<(), Self::Err>;
}

pub trait TensorToArray<S: Shape, E: Unit>: DeviceStorage {
    type Array: std::fmt::Debug + PartialEq;
    fn tensor_to_array<T>(&self, tensor: &Tensor<S, E, Self, T>) -> Self::Array;
}

pub trait AsArray {
    type Array: std::fmt::Debug + PartialEq;
    fn array(&self) -> Self::Array;
}

impl<S: Shape, E: Unit, D: TensorToArray<S, E>, T> AsArray for Tensor<S, E, D, T> {
    type Array = D::Array;
    /// Convert tensors to rust arrays
    fn array(&self) -> Self::Array {
        self.device.tensor_to_array(self)
    }
}

impl<S: Shape, E: Unit, D: DeviceStorage, T> Tensor<S, E, D, T> {
    pub fn as_vec(&self) -> std::vec::Vec<E> {
        self.device.tensor_to_vec(self)
    }
}

/// Construct tensors from rust vectors. This trait is only used to implement TensorFrom.
pub trait TensorFromVec<E: Unit>: DeviceStorage {
    fn tensor_from_vec<S: Shape>(&self, src: Vec<E>, shape: S) -> Tensor<S, E, Self> {
        self.try_tensor_from_vec::<S>(src, shape).unwrap()
    }

    fn try_tensor_from_vec<S: Shape>(
        &self,
        src: Vec<E>,
        shape: S,
    ) -> Result<Tensor<S, E, Self>, Self::Err>;
}

/// Construct tensors from rust data
pub trait TensorFrom<Src, S: Shape, E: Unit>: DeviceStorage {
    /// Create a tensor from rust data
    /// ```rust
    /// # use dfdx::prelude::*;
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
    fn try_tensor(&self, src: Src) -> Result<Tensor<S, E, Self>, Self::Err>;
}

impl<E: Unit, D: DeviceStorage + TensorFromVec<E>> TensorFrom<E, Rank0, E> for D {
    fn try_tensor(&self, src: E) -> Result<Tensor<Rank0, E, Self>, Self::Err> {
        self.try_tensor_from_vec(vec![src], ())
    }
}

impl<E: Unit, const M: usize, D: DeviceStorage + TensorFromVec<E>> TensorFrom<[E; M], Rank1<M>, E>
    for D
{
    fn try_tensor(&self, src: [E; M]) -> Result<Tensor<Rank1<M>, E, Self>, Self::Err> {
        self.try_tensor(&src)
    }
}

impl<E: Unit, const M: usize, D: DeviceStorage + TensorFromVec<E>> TensorFrom<&[E; M], Rank1<M>, E>
    for D
{
    fn try_tensor(&self, src: &[E; M]) -> Result<Tensor<Rank1<M>, E, Self>, Self::Err> {
        self.try_tensor_from_vec(src.to_vec(), (Const::<M>,))
    }
}

impl<E: Unit, const M: usize, const N: usize, D> TensorFrom<[[E; N]; M], Rank2<M, N>, E> for D
where
    D: DeviceStorage + TensorFromVec<E>,
{
    fn try_tensor(&self, src: [[E; N]; M]) -> Result<Tensor<Rank2<M, N>, E, Self>, Self::Err> {
        let vec: Vec<E> = src.iter().flat_map(|v| v.iter().copied()).collect();

        self.try_tensor_from_vec(vec, (Const::<M>, Const::<N>))
    }
}

impl<E: Unit, const M: usize, const N: usize, const O: usize, D>
    TensorFrom<[[[E; O]; N]; M], Rank3<M, N, O>, E> for D
where
    D: DeviceStorage + TensorFromVec<E>,
{
    fn try_tensor(
        &self,
        src: [[[E; O]; N]; M],
    ) -> Result<Tensor<Rank3<M, N, O>, E, Self>, Self::Err> {
        let vec: Vec<E> = src
            .iter()
            .flat_map(|v| v.iter())
            .flat_map(|v| v.iter().copied())
            .collect();

        self.try_tensor_from_vec(vec, (Const::<M>, Const::<N>, Const::<O>))
    }
}

impl<E: Unit, const M: usize, const N: usize, const O: usize, const P: usize, D>
    TensorFrom<[[[[E; P]; O]; N]; M], Rank4<M, N, O, P>, E> for D
where
    D: DeviceStorage + TensorFromVec<E>,
{
    fn try_tensor(
        &self,
        src: [[[[E; P]; O]; N]; M],
    ) -> Result<Tensor<Rank4<M, N, O, P>, E, Self>, Self::Err> {
        let vec: Vec<E> = src
            .iter()
            .flat_map(|v| v.iter())
            .flat_map(|v| v.iter())
            .flat_map(|v| v.iter().copied())
            .collect();

        self.try_tensor_from_vec(vec, (Const::<M>, Const::<N>, Const::<O>, Const::<P>))
    }
}

impl<E: Unit, S: ConstShape, D: DeviceStorage + TensorFromVec<E>> TensorFrom<Vec<E>, S, E> for D {
    fn try_tensor(&self, src: Vec<E>) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_tensor_from_vec(src, S::default())
    }
}

impl<E: Unit, S: Shape, D: DeviceStorage + TensorFromVec<E>> TensorFrom<(Vec<E>, S), S, E> for D {
    fn try_tensor(&self, (src, shape): (Vec<E>, S)) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_tensor_from_vec(src, shape)
    }
}
