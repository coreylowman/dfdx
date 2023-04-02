use rand::distributions::Distribution;
use rand_distr::{Standard, StandardNormal};
use std::vec::Vec;

use crate::shapes::*;

use super::{cpu::LendingIterator, Tensor};

/// Represents something that has an error associated type
pub trait HasErr: Sized {
    type Err: std::fmt::Debug + std::fmt::Display;
}

/// Convert tensors to [std::vec::Vec]
pub trait AsVec<E> {
    fn as_vec(&self) -> std::vec::Vec<E>;
}

#[derive(Copy, Clone, Debug, Default)]
pub struct VecStorage;

pub trait DeviceStorage<E: Unit>: Clone {
    type Storage: Storage<E>;
}

impl<E: Unit> DeviceStorage<E> for VecStorage {
    type Storage = Vec<E>;
}

pub trait Storage<E: Unit>: std::fmt::Debug + Clone + Send + Sync {
    type Iter<'a>: LendingIterator<Item<'a> = &'a E>
    where
        Self: 'a;
    type IterMut<'a>: LendingIterator<Item<'a> = &'a mut E>
    where
        Self: 'a;
    type Err;

    fn try_alloc_zeros(numel: usize) -> Result<Self, Self::Err> {
        Self::try_alloc_elem(numel, Default::default())
    }

    fn try_alloc_elem(numel: usize, elem: E) -> Result<Self, Self::Err>;

    fn from_vec(vec: Vec<E>) -> Self;

    fn fill(&mut self, value: E);

    fn index(&self, index: usize) -> E;

    fn iter(&self) -> Self::Iter<'_>;

    fn iter_mut(&mut self) -> Self::IterMut<'_>;

    fn len(&self) -> usize;
}

impl<E: Unit> Storage<E> for Vec<E> {
    type Iter<'a> = std::slice::Iter<'a, E>;
    type IterMut<'a> = std::slice::IterMut<'a, E>;
    type Err = std::collections::TryReserveError;

    #[inline]
    fn try_alloc_elem(numel: usize, elem: E) -> Result<Self, Self::Err> {
        #[cfg(feature = "fast-alloc")]
        {
            Ok(std::vec![elem; numel])
        }

        #[cfg(not(feature = "fast-alloc"))]
        {
            let mut data = Self::new();
            data.try_reserve(numel).map_err(|_| CpuError::OutOfMemory)?;
            data.resize(numel, elem);
            Ok(data)
        }
    }

    fn from_vec(vec: Vec<E>) -> Self {
        vec
    }

    fn fill(&mut self, value: E) {
        self.as_mut_slice().fill(value)
    }

    fn index(&self, index: usize) -> E {
        self[index]
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.as_slice().iter()
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.as_mut_slice().iter_mut()
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

pub trait RandomU64 {
    fn random_u64(&self) -> u64;
}

pub trait DeviceAllocGrad<E: Unit>: 'static + DeviceStorage<E> + HasErr {
    /// Allocates a gradient for the given nd array
    fn try_alloc_grad(&self, storage: &Self::Storage) -> Result<Self::Storage, Self::Err>;
}

pub trait DeviceTensorToVec<E: Unit>: DeviceStorage<E> + HasErr {
    fn tensor_to_vec<S: Shape, T>(&self, tensor: &Tensor<S, E, Self, T>) -> Vec<E>;
}

/// Internal trait - Represents something that can allocate its own gradient.
pub trait AllocGrad: HasErr {
    type Gradient;
    fn try_alloc_grad(&self) -> Result<Self::Gradient, Self::Err>;
}

impl<S: Shape, E: Unit, D: DeviceAllocGrad<E>, T> AllocGrad for Tensor<S, E, D, T> {
    type Gradient = D::Storage;
    fn try_alloc_grad(&self) -> Result<Self::Gradient, D::Err> {
        self.device.try_alloc_grad(self.data.as_ref())
    }
}

/// Enables copying data into and out of tensors
pub trait CopySlice<E: Unit>: DeviceStorage<E> + Sized {
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
pub trait ZerosTensor<E: Unit>: DeviceStorage<E> + HasErr {
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

pub trait ZeroFillStorage<E: Unit>: DeviceStorage<E> + HasErr {
    fn try_fill_with_zeros(&self, storage: &mut Self::Storage) -> Result<(), Self::Err>;
}

/// Construct tensors filled with ones.
pub trait OnesTensor<E: Unit>: DeviceStorage<E> + HasErr {
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

pub trait OneFillStorage<E: Unit>: DeviceStorage<E> + HasErr {
    fn try_fill_with_ones(&self, storage: &mut Self::Storage) -> Result<(), Self::Err>;
}

/// Build upper & lower triangle tensors.
pub trait TriangleTensor<E: Unit>: DeviceStorage<E> + HasErr {
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
pub trait SampleTensor<E: Unit>: DeviceStorage<E> + HasErr {
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
        storage: &mut Self::Storage,
        distr: D,
    ) -> Result<(), Self::Err>;
}

pub trait TensorToArray<E: Unit, S: Shape>: DeviceStorage<E> + HasErr {
    type Array: std::fmt::Debug + PartialEq;
    fn tensor_to_array<T>(&self, tensor: &Tensor<S, E, Self, T>) -> Self::Array;
}

pub trait AsArray<E> {
    type Array: std::fmt::Debug + PartialEq;
    fn array(&self) -> Self::Array;
}

impl<S: Shape, E: Unit, D: TensorToArray<E, S>, T> AsArray<E> for Tensor<S, E, D, T> {
    type Array = D::Array;
    /// Convert tensors to rust arrays
    fn array(&self) -> Self::Array {
        self.device.tensor_to_array(self)
    }
}

impl<S: Shape, E: Unit, D: DeviceTensorToVec<E>, T> Tensor<S, E, D, T> {
    pub fn as_vec(&self) -> std::vec::Vec<E> {
        self.device.tensor_to_vec(self)
    }
}

/// Construct tensors from rust vectors. This trait is only used to implement TensorFrom.
pub trait TensorFromVec<E: Unit>: DeviceStorage<E> + HasErr {
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
pub trait TensorFrom<Src, S: Shape, E: Unit>: DeviceStorage<E> + HasErr {
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

impl<E: Unit, D: TensorFromVec<E>> TensorFrom<E, Rank0, E> for D {
    fn try_tensor(&self, src: E) -> Result<Tensor<Rank0, E, Self>, Self::Err> {
        self.try_tensor_from_vec(vec![src], ())
    }
}

impl<E: Unit, const M: usize, D: TensorFromVec<E>> TensorFrom<[E; M], Rank1<M>, E> for D {
    fn try_tensor(&self, src: [E; M]) -> Result<Tensor<Rank1<M>, E, Self>, Self::Err> {
        self.try_tensor(&src)
    }
}

impl<E: Unit, const M: usize, D: TensorFromVec<E>> TensorFrom<&[E; M], Rank1<M>, E> for D {
    fn try_tensor(&self, src: &[E; M]) -> Result<Tensor<Rank1<M>, E, Self>, Self::Err> {
        self.try_tensor_from_vec(src.to_vec(), (Const::<M>,))
    }
}

impl<E: Unit, const M: usize, const N: usize, D> TensorFrom<[[E; N]; M], Rank2<M, N>, E> for D
where
    D: TensorFromVec<E>,
{
    fn try_tensor(&self, src: [[E; N]; M]) -> Result<Tensor<Rank2<M, N>, E, Self>, Self::Err> {
        let vec: Vec<E> = src.iter().flat_map(|v| v.iter().copied()).collect();

        self.try_tensor_from_vec(vec, (Const::<M>, Const::<N>))
    }
}

impl<E: Unit, const M: usize, const N: usize, const O: usize, D>
    TensorFrom<[[[E; O]; N]; M], Rank3<M, N, O>, E> for D
where
    D: TensorFromVec<E>,
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
    D: TensorFromVec<E>,
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

impl<E: Unit, S: ConstShape, D: TensorFromVec<E>> TensorFrom<Vec<E>, S, E> for D {
    fn try_tensor(&self, src: Vec<E>) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_tensor_from_vec(src, S::default())
    }
}

impl<E: Unit, S: Shape, D: TensorFromVec<E>> TensorFrom<(Vec<E>, S), S, E> for D {
    fn try_tensor(&self, (src, shape): (Vec<E>, S)) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_tensor_from_vec(src, shape)
    }
}
