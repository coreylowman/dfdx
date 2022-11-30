use crate::{
    arrays::{Dtype, HasDtype, HasShape, Shape, TryFromNumElements},
    unique_id::unique_id,
};

use super::Tensor;

pub trait HasErr: Sized {
    type Err: std::fmt::Debug + std::fmt::Display;
}

pub trait DeviceStorage: 'static + Default + Clone + HasErr {
    type Storage<S: Shape, Elem: Dtype>: 'static
        + std::fmt::Debug
        + Clone
        + Send
        + Sync
        + HasShape<Shape = S>;

    fn random_u64(&self) -> u64;
    fn try_alloc<S: Shape, E: Dtype>(&self, shape: &S) -> Result<Self::Storage<S, E>, Self::Err>;
    fn try_alloc_like<S: Shape, E: Dtype>(
        &self,
        storage: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err>;
}

pub trait TensorFromStorage: DeviceStorage {
    fn upgrade<S: Shape, E: Dtype>(&self, storage: Self::Storage<S, E>) -> Tensor<S, E, Self>;
}

impl<D: DeviceStorage> TensorFromStorage for D {
    fn upgrade<S: Shape, E: Dtype>(&self, storage: Self::Storage<S, E>) -> Tensor<S, E, Self> {
        Tensor {
            id: unique_id(),
            storage,
            device: self.clone(),
            tape: Default::default(),
        }
    }
}

pub trait AllocOn<D: DeviceStorage>: HasShape + HasDtype {
    fn try_alloc_like(&self) -> Result<D::Storage<Self::Shape, Self::Dtype>, D::Err>;
}

pub trait ZerosTensor<E: Dtype>: DeviceStorage {
    fn zeros<S: Shape + Default>(&self) -> Tensor<S, E, Self> {
        self.try_zeros_like::<S>(&Default::default()).unwrap()
    }
    fn try_zeros<S: Shape + Default>(&self) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_zeros_like::<S>(&Default::default())
    }
    fn zeros_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self> {
        self.try_zeros_like(src).unwrap()
    }
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;
}

pub trait ZeroFillStorage<E: Dtype>: DeviceStorage {
    fn try_fill_with_zeros<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, E>,
    ) -> Result<(), Self::Err>;
}

pub trait OnesTensor<E: Dtype>: DeviceStorage {
    fn ones<S: Shape + Default>(&self) -> Tensor<S, E, Self> {
        self.try_ones_like::<S>(&Default::default()).unwrap()
    }
    fn try_ones<S: Shape + Default>(&self) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_ones_like::<S>(&Default::default())
    }
    fn ones_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self> {
        self.try_ones_like(src).unwrap()
    }
    fn try_ones_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;
}

pub trait OneFillStorage<E: Dtype>: DeviceStorage {
    fn try_fill_with_ones<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, E>,
    ) -> Result<(), Self::Err>;
}

pub trait RandTensor<E: Dtype>: DeviceStorage {
    fn rand<S: Shape + Default>(&self) -> Tensor<S, E, Self> {
        self.try_rand_like::<S>(&Default::default()).unwrap()
    }
    fn try_rand<S: Shape + Default>(&self) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_rand_like::<S>(&Default::default())
    }
    fn rand_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self> {
        self.try_rand_like(src).unwrap()
    }
    fn try_rand_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;

    fn try_uniform<S: Shape + Default>(
        &self,
        min: E,
        max: E,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_uniform_like::<S>(&Default::default(), min, max)
    }
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

pub trait RandnTensor<E: Dtype>: DeviceStorage {
    fn randn<S: Shape + Default>(&self) -> Tensor<S, E, Self> {
        self.try_randn_like::<S>(&Default::default()).unwrap()
    }
    fn try_randn<S: Shape + Default>(&self) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_randn_like::<S>(&Default::default())
    }
    fn randn_like<S: HasShape>(&self, src: &S) -> Tensor<S::Shape, E, Self> {
        self.try_randn_like(src).unwrap()
    }
    fn try_randn_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;
    fn try_normal<S: Shape + Default>(
        &self,
        mean: E,
        stddev: E,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_normal_like::<S>(&Default::default(), mean, stddev)
    }
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

pub trait TensorFromVec<E: Dtype>: DeviceStorage {
    fn from_vec<S: Shape + TryFromNumElements>(
        &self,
        src: std::vec::Vec<E>,
    ) -> Option<Tensor<S, E, Self>> {
        self.try_from_vec(src).map(|t| t.unwrap())
    }

    fn try_from_vec<S: Shape + TryFromNumElements>(
        &self,
        src: std::vec::Vec<E>,
    ) -> Option<Result<Tensor<S, E, Self>, Self::Err>>;
}

pub trait TensorFromSlice<E: Dtype>: DeviceStorage {
    fn from_slice<S: Shape + TryFromNumElements>(&self, src: &[E]) -> Option<Tensor<S, E, Self>> {
        self.try_from_slice(src).map(|t| t.unwrap())
    }
    fn try_from_slice<S: Shape + TryFromNumElements>(
        &self,
        src: &[E],
    ) -> Option<Result<Tensor<S, E, Self>, Self::Err>>;
}

pub trait TensorFromArray<Src, S: Shape, E: Dtype>: DeviceStorage {
    fn tensor(&self, src: Src) -> Tensor<S, E, Self> {
        self.try_from_array(src).unwrap()
    }
    fn try_tensor(&self, src: Src) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_from_array(src)
    }
    fn from_array(&self, src: Src) -> Tensor<S, E, Self> {
        self.try_from_array(src).unwrap()
    }
    fn try_from_array(&self, src: Src) -> Result<Tensor<S, E, Self>, Self::Err>;
}

pub trait AsArray {
    type Array;
    fn as_array(&self) -> Self::Array;
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T> AsVec for Tensor<S, E, D, T>
where
    D::Storage<S, E>: AsVec,
{
    type Vec = <D::Storage<S, E> as AsVec>::Vec;
    fn as_vec(&self) -> Self::Vec {
        self.storage.as_vec()
    }
}

pub trait AsVec {
    type Vec;
    fn as_vec(&self) -> Self::Vec;
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T> AsArray for Tensor<S, E, D, T>
where
    D::Storage<S, E>: AsArray,
{
    type Array = <D::Storage<S, E> as AsArray>::Array;
    fn as_array(&self) -> Self::Array {
        self.storage.as_array()
    }
}
