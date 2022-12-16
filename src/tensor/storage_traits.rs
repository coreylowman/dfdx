use crate::{
    shapes::{Dtype, HasDtype, HasShape, Shape},
    unique_id::unique_id,
};

use super::Tensor;

/// Represents something that has an error associated type
pub trait HasErr: Sized {
    type Err: std::fmt::Debug + std::fmt::Display;
}

/// Something that can store nd arrays for a given [Shape] and [Dtype]
pub trait DeviceStorage: 'static + Default + Clone + HasErr {
    /// Generic storage type
    type Storage<S: Shape, E: Dtype>: 'static
        + std::fmt::Debug
        + Clone
        + Send
        + Sync
        + HasShape<Shape = S>;

    /// Generates a random u64 number
    fn random_u64(&self) -> u64;

    /// Allocates a gradient for the given nd array
    fn try_alloc_grad<S: Shape, E: Dtype>(
        &self,
        storage: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err>;

    /// Upgrades the device storage into a tensor
    fn upgrade<S: Shape, E: Dtype>(&self, storage: Self::Storage<S, E>) -> Tensor<S, E, Self> {
        Tensor {
            id: unique_id(),
            storage,
            device: self.clone(),
            tape: Default::default(),
        }
    }
}

/// Internal trait - Represents something that can allocate its own gradient.
pub trait AllocGrad<D: DeviceStorage>: HasShape + HasDtype {
    fn try_alloc_grad(&self) -> Result<D::Storage<Self::Shape, Self::Dtype>, D::Err>;
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T> AllocGrad<D> for Tensor<S, E, D, T> {
    fn try_alloc_grad(&self) -> Result<D::Storage<Self::Shape, Self::Dtype>, D::Err> {
        self.device.try_alloc_grad(&self.storage)
    }
}

/// Enables copying data into and out of tensors
pub trait CopySlice<E: Dtype>: DeviceStorage {
    fn copy_from<S: Shape, T>(dst: &mut Tensor<S, E, Self, T>, src: &[E]);
    fn copy_into<S: Shape, T>(src: &Tensor<S, E, Self, T>, dst: &mut [E]);
}

impl<S: Shape, E: Dtype, D: CopySlice<E>, T> Tensor<S, E, D, T> {
    /// Copy data from a slice - **panics** if there are not enough elements in the slice.
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

    /// Copy data into a slice - **panics** if there are not enough elements in the tensor.
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
pub trait ZerosTensor<E: Dtype>: DeviceStorage {
    /// Creates a tensor filled with zeros.
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32> = dev.zeros();
    /// ```
    fn zeros<S: Shape + Default>(&self) -> Tensor<S, E, Self> {
        self.try_zeros_like::<S>(&Default::default()).unwrap()
    }

    /// Fallible version of [ZerosTensor::zeros]
    fn try_zeros<S: Shape + Default>(&self) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_zeros_like::<S>(&Default::default())
    }

    /// Build the tensor with a shape given by something else.
    ///
    /// Given a shape directly:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<(usize, Const<3>), f32> = dev.zeros_like(&(5, Const));
    /// ```
    ///
    /// Given another tensor:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32> = dev.zeros();
    /// let b: Tensor<Rank2<2, 3>, f32> = dev.zeros_like(&a);
    /// ```
    fn zeros_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self> {
        self.try_zeros_like(src).unwrap()
    }

    /// Fallible version of [ZerosTensor::zeros_like]
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;
}

pub trait ZeroFillStorage<E: Dtype>: DeviceStorage {
    fn try_fill_with_zeros<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, E>,
    ) -> Result<(), Self::Err>;
}

/// Construct tensors filled with ones.
pub trait OnesTensor<E: Dtype>: DeviceStorage {
    /// Creates a tensor filled with ones.
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32> = dev.ones();
    /// ```
    fn ones<S: Shape + Default>(&self) -> Tensor<S, E, Self> {
        self.try_ones_like::<S>(&Default::default()).unwrap()
    }

    /// Fallible version of [OnesTensor::ones]
    fn try_ones<S: Shape + Default>(&self) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_ones_like::<S>(&Default::default())
    }

    /// Build the tensor with a shape given by something else.
    ///
    /// Given a shape directly:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<(usize, Const<3>), f32> = dev.ones_like(&(5, Const));
    /// ```
    ///
    /// Given another tensor:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32> = dev.ones();
    /// let b = dev.ones_like(&a);
    /// ```
    fn ones_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self> {
        self.try_ones_like(src).unwrap()
    }

    /// Fallible version of [OnesTensor::ones_like]
    fn try_ones_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;
}

pub trait OneFillStorage<E: Dtype>: DeviceStorage {
    fn try_fill_with_ones<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, E>,
    ) -> Result<(), Self::Err>;
}

/// Constructs tensors filled with random values from a uniform distribution.
pub trait RandTensor<E: Dtype>: DeviceStorage {
    /// Creates a tensor filled with random data in the range [0.0, 1.0).
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32> = dev.rand();
    /// ```
    fn rand<S: Shape + Default>(&self) -> Tensor<S, E, Self> {
        self.try_rand_like::<S>(&Default::default()).unwrap()
    }
    /// Fallible version of [RandTensor::rand]
    fn try_rand<S: Shape + Default>(&self) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_rand_like::<S>(&Default::default())
    }
    /// Build the tensor with a shape given by something else.
    ///
    /// Given a shape directly:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<(usize, Const<3>), f32> = dev.rand_like(&(5, Const::<3>));
    /// ```
    ///
    /// Given another tensor:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32> = dev.rand();
    /// let b = dev.rand_like(&a);
    /// ```
    fn rand_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self> {
        self.try_rand_like(src).unwrap()
    }
    /// Fallible version of [RandTensor::rand_like]
    fn try_rand_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;

    /// Creates a tensor filled with random data in a custom range.
    fn try_uniform<S: Shape + Default>(
        &self,
        min: E,
        max: E,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_uniform_like::<S>(&Default::default(), min, max)
    }
    /// Creates a tensor filled with random data in a custom range, using the shape of
    /// something else.
    fn try_uniform_like<S: HasShape>(
        &self,
        src: &S,
        min: E,
        max: E,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;
}

pub trait RandFillStorage<E: Dtype>: DeviceStorage {
    fn try_fill_with_uniform<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, E>,
        min: E,
        max: E,
    ) -> Result<(), Self::Err>;
}

/// Construct tensors with random values from a normal distribution
pub trait RandnTensor<E: Dtype>: DeviceStorage {
    /// Creates a tensor filled with random data from a normal distribution with mean 0 and stddev 1.
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32> = dev.randn();
    /// ```
    fn randn<S: Shape + Default>(&self) -> Tensor<S, E, Self> {
        self.try_randn_like::<S>(&Default::default()).unwrap()
    }
    /// Fallible version of [RandnTensor::randn]
    fn try_randn<S: Shape + Default>(&self) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_randn_like::<S>(&Default::default())
    }
    /// Build the tensor with a shape given by something else.
    ///
    /// Given a shape directly:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<(usize, Const<3>), f32> = dev.randn_like(&(5, Const::<3>));
    /// ```
    ///
    /// Given another tensor:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32> = dev.randn();
    /// let b = dev.randn_like(&a);
    /// ```
    fn randn_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self> {
        self.try_randn_like(src).unwrap()
    }
    /// Fallible version of [RandnTensor::randn_like]
    fn try_randn_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;
    /// Creates a tensor filled with random data with custom parameters.
    fn try_normal<S: Shape + Default>(
        &self,
        mean: E,
        stddev: E,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_normal_like::<S>(&Default::default(), mean, stddev)
    }
    /// Creates a tensor filled with random data with custom parameters, using the shape of
    /// something else.
    fn try_normal_like<S: HasShape>(
        &self,
        src: &S,
        mean: E,
        stddev: E,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;
}

pub trait RandnFillStorage<E: Dtype>: DeviceStorage {
    fn try_fill_with_normal<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, E>,
        mean: E,
        stddev: E,
    ) -> Result<(), Self::Err>;
}

/// Construct tensors from rust arrays
pub trait TensorFromArray<Src, S: Shape, E: Dtype>: DeviceStorage {
    /// Create a tensor from a rust array
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let _: Tensor<Rank2<2, 3>, f32> = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// ```
    fn tensor(&self, src: Src) -> Tensor<S, E, Self> {
        self.try_tensor(src).unwrap()
    }
    /// Fallible version of [TensorFromArray::tensor]
    fn try_tensor(&self, src: Src) -> Result<Tensor<S, E, Self>, Self::Err>;
}

/// Convert tensors to rust arrays
pub trait AsArray {
    type Array: std::fmt::Debug;
    fn array(&self) -> Self::Array;
}
impl<S: Shape, E: Dtype, D: DeviceStorage, T> AsArray for Tensor<S, E, D, T>
where
    D::Storage<S, E>: AsArray,
{
    type Array = <D::Storage<S, E> as AsArray>::Array;
    fn array(&self) -> Self::Array {
        self.storage.array()
    }
}

/// Convert tensors to [std::vec::Vec]
pub trait AsVec: HasDtype {
    fn as_vec(&self) -> std::vec::Vec<Self::Dtype>;
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T> AsVec for Tensor<S, E, D, T>
where
    D::Storage<S, E>: HasDtype<Dtype = E> + AsVec,
{
    fn as_vec(&self) -> std::vec::Vec<E> {
        self.storage.as_vec()
    }
}
