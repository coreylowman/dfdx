use super::storage_traits::{DeviceStorage, HasErr};
use super::{Cpu, OneFillStorage, RandFillStorage, RandnFillStorage, ZeroFillStorage};
use crate::{
    gradients::{NoneTape, OwnedTape, Tape},
    shapes::*,
    unique_id::{HasUniqueId, UniqueId},
};

#[derive(Debug, Clone)]
pub struct Tensor<S: Shape, E: Dtype, D: DeviceStorage = Cpu, T = NoneTape> {
    pub(crate) id: UniqueId,
    pub(crate) storage: D::Storage<S, E>,
    pub(crate) device: D,
    pub(crate) tape: T,
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T> HasShape for Tensor<S, E, D, T> {
    type WithShape<New: Shape> = Tensor<New, E, D, T>;
    type Shape = S;
    fn shape(&self) -> &Self::Shape {
        self.storage.shape()
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T> HasDtype for Tensor<S, E, D, T> {
    type Dtype = E;
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T> HasUniqueId for Tensor<S, E, D, T> {
    fn id(&self) -> &UniqueId {
        &self.id
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T> HasErr for Tensor<S, E, D, T> {
    type Err = D::Err;
}

impl<S: Shape, E: Dtype, D: DeviceStorage> Tensor<S, E, D, NoneTape> {
    pub fn trace(&self) -> Tensor<S, E, D, OwnedTape<D>> {
        self.clone().traced()
    }

    pub fn traced(self) -> Tensor<S, E, D, OwnedTape<D>> {
        self.put_tape(Default::default())
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn retaped<New: Tape<D>>(&self) -> Tensor<S, E, D, New> {
        Tensor {
            id: self.id,
            storage: self.storage.clone(),
            device: self.device.clone(),
            tape: Default::default(),
        }
    }
}

pub trait PutTape<T> {
    type Output;
    fn put_tape(self, tape: T) -> Self::Output;
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T> PutTape<T> for Tensor<S, E, D> {
    type Output = Tensor<S, E, D, T>;
    fn put_tape(self, tape: T) -> Self::Output {
        Tensor {
            id: self.id,
            storage: self.storage,
            device: self.device,
            tape,
        }
    }
}

pub trait SplitTape {
    type Tape: Default;
    type NoTape: Clone + PutTape<Self::Tape, Output = Self>;
    fn split_tape(self) -> (Self::NoTape, Self::Tape);
    fn with_empty_tape(&self) -> Self;
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T: Default> SplitTape for Tensor<S, E, D, T> {
    type Tape = T;
    type NoTape = Tensor<S, E, D>;
    fn split_tape(self) -> (Self::NoTape, Self::Tape) {
        (
            Tensor {
                id: self.id,
                storage: self.storage,
                device: self.device,
                tape: NoneTape,
            },
            self.tape,
        )
    }

    fn with_empty_tape(&self) -> Self {
        Self {
            id: self.id,
            storage: self.storage.clone(),
            device: self.device.clone(),
            tape: Default::default(),
        }
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage + ZeroFillStorage<E>, T> Tensor<S, E, D, T> {
    pub fn fill_with_zeros(&mut self) {
        self.try_fill_with_zeros().unwrap()
    }

    pub fn try_fill_with_zeros(&mut self) -> Result<(), D::Err> {
        self.device.try_fill_with_zeros(&mut self.storage)
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage + OneFillStorage<E>, T> Tensor<S, E, D, T> {
    pub fn fill_with_ones(&mut self) {
        self.try_fill_with_ones().unwrap()
    }

    pub fn try_fill_with_ones(&mut self) -> Result<(), D::Err> {
        self.device.try_fill_with_ones(&mut self.storage)
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage + RandFillStorage<E>, T> Tensor<S, E, D, T> {
    pub fn fill_with_uniform(&mut self, min: E, max: E) {
        self.try_fill_with_uniform(min, max).unwrap()
    }

    pub fn try_fill_with_uniform(&mut self, min: E, max: E) -> Result<(), D::Err> {
        self.device
            .try_fill_with_uniform(&mut self.storage, min, max)
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage + RandnFillStorage<E>, T> Tensor<S, E, D, T> {
    pub fn fill_with_normal(&mut self, mean: E, stddev: E) {
        self.try_fill_with_normal(mean, stddev).unwrap()
    }

    pub fn try_fill_with_normal(&mut self, mean: E, stddev: E) -> Result<(), D::Err> {
        self.device
            .try_fill_with_normal(&mut self.storage, mean, stddev)
    }
}

pub type Tensor0D<Tape = NoneTape> = Tensor<Rank0, f32, Cpu, Tape>;
pub type Tensor1D<const M: usize, Tape = NoneTape> = Tensor<Rank1<M>, f32, Cpu, Tape>;
pub type Tensor2D<const M: usize, const N: usize, Tape = NoneTape> =
    Tensor<Rank2<M, N>, f32, Cpu, Tape>;
pub type Tensor3D<const M: usize, const N: usize, const O: usize, D, Tape = NoneTape> =
    Tensor<Rank3<M, N, O>, f32, D, Tape>;
pub type Tensor4D<const M: usize, const N: usize, const O: usize, const P: usize, Tape = NoneTape> =
    Tensor<Rank4<M, N, O, P>, f32, Cpu, Tape>;
pub type Tensor5D<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
    Tape = NoneTape,
> = Tensor<Rank5<M, N, O, P, Q>, f32, Cpu, Tape>;
pub type Tensor6D<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
    const R: usize,
    Tape = NoneTape,
> = Tensor<Rank6<M, N, O, P, Q, R>, f32, Cpu, Tape>;
