use super::base::Tensor;
use crate::arrays::{Dtype, Shape};
use crate::devices::{
    AsArray, AsVec, Device, Ones, OnesLike, Rand, RandLike, Randn, RandnLike, TryConvert, Zeros,
    ZerosLike,
};
use crate::gradients::NoneTape;
use crate::unique_id::unique_id;

pub(crate) fn make_tensor<S: Shape, E: Dtype, D: Device>(
    device: &D,
    storage: D::Storage<S, E>,
) -> Tensor<S, E, D, NoneTape> {
    Tensor {
        id: unique_id(),
        storage,
        device: device.clone(),
        tape: NoneTape,
    }
}

pub trait TensorSugar<Src, S: Shape, E: Dtype>: Device {
    fn tensor(&self, src: Src) -> Tensor<S, E, Self, NoneTape>;
}

impl<Src, S: Shape, E: Dtype, D: Device> TensorSugar<Src, S, E> for D
where
    D: TryConvert<Src, Tensor<S, E, D, NoneTape>>,
{
    fn tensor(&self, src: Src) -> Tensor<S, E, Self, NoneTape> {
        self.convert(src)
    }
}

impl<S: Shape, E: Dtype, D: Device> Zeros<Tensor<S, E, D, NoneTape>> for D
where
    Self: Zeros<D::Storage<S, E>>,
{
    fn try_zeros(&self) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(make_tensor(self, self.try_zeros()?))
    }
    fn fill_with_zeros(&self, t: &mut Tensor<S, E, D, NoneTape>) {
        self.fill_with_zeros(&mut t.storage);
    }
}

impl<S: Shape, E: Dtype, D: Device> ZerosLike<S, Tensor<S, E, D, NoneTape>> for D
where
    Self: ZerosLike<S, D::Storage<S, E>>,
{
    fn try_zeros_like(&self, src: S) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(make_tensor(self, self.try_zeros_like(src)?))
    }
}

impl<S: Shape, E: Dtype, D: Device> Ones<Tensor<S, E, D, NoneTape>> for D
where
    Self: Ones<D::Storage<S, E>>,
{
    fn try_ones(&self) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(make_tensor(self, self.try_ones()?))
    }
    fn fill_with_ones(&self, t: &mut Tensor<S, E, D, NoneTape>) {
        self.fill_with_ones(&mut t.storage);
    }
}

impl<S: Shape, E: Dtype, D: Device> OnesLike<S, Tensor<S, E, D, NoneTape>> for D
where
    Self: OnesLike<S, D::Storage<S, E>>,
{
    fn try_ones_like(&self, src: S) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(make_tensor(self, self.try_ones_like(src)?))
    }
}

impl<S: Shape, E: Dtype, D: Device> Rand<Tensor<S, E, D, NoneTape>> for D
where
    Self: Rand<D::Storage<S, E>>,
{
    fn try_rand(&self) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(make_tensor(self, self.try_rand()?))
    }
    fn fill_with_rand(&self, t: &mut Tensor<S, E, D, NoneTape>) {
        self.fill_with_rand(&mut t.storage);
    }
}

impl<S: Shape, E: Dtype, D: Device> RandLike<S, Tensor<S, E, D, NoneTape>> for D
where
    Self: RandLike<S, D::Storage<S, E>>,
{
    fn try_rand_like(&self, src: S) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(make_tensor(self, self.try_rand_like(src)?))
    }
}

impl<S: Shape, E: Dtype, D: Device> Randn<Tensor<S, E, D, NoneTape>> for D
where
    Self: Randn<D::Storage<S, E>>,
{
    fn try_randn(&self) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(make_tensor(self, self.try_randn()?))
    }
    fn fill_with_randn(&self, t: &mut Tensor<S, E, D, NoneTape>) {
        self.fill_with_randn(&mut t.storage);
    }
}

impl<S: Shape, E: Dtype, D: Device> RandnLike<S, Tensor<S, E, D, NoneTape>> for D
where
    Self: RandnLike<S, D::Storage<S, E>>,
{
    fn try_randn_like(&self, src: S) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(make_tensor(self, self.try_randn_like(src)?))
    }
}

impl<S: Shape, Src, E: Dtype, D: Device> TryConvert<Src, Tensor<S, E, D, NoneTape>> for D
where
    Self: TryConvert<Src, D::Storage<S, E>>,
{
    fn try_convert(&self, src: Src) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(make_tensor(self, self.try_convert(src)?))
    }
}

impl<S: Shape, E: Dtype, D: Device, T> AsVec for Tensor<S, E, D, T>
where
    D::Storage<S, E>: AsVec,
{
    type Vec = <D::Storage<S, E> as AsVec>::Vec;
    fn as_vec(&self) -> Self::Vec {
        self.storage.as_vec()
    }
}

impl<S: Shape, E: Dtype, D: Device, T> AsArray for Tensor<S, E, D, T>
where
    D::Storage<S, E>: AsArray,
{
    type Array = <D::Storage<S, E> as AsArray>::Array;
    fn as_array(&self) -> Self::Array {
        self.storage.as_array()
    }
}
