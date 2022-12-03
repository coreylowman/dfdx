use super::storage::{AllocGradOn, DeviceStorage, HasErr};
use super::{OneFillStorage, RandFillStorage, RandnFillStorage, ZeroFillStorage};
use crate::arrays::{
    Dtype, HasDtype, HasShape, Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Shape,
};
use crate::unique_id::HasUniqueId;
use crate::{
    gradients::{NoneTape, OwnedTape, Tape},
    unique_id::UniqueId,
};

#[derive(Debug, Clone)]
pub struct Tensor<S: Shape, E: Dtype, D: DeviceStorage, T = NoneTape> {
    pub(crate) id: UniqueId,
    pub(crate) storage: D::Storage<S, E>,
    pub(crate) device: D,
    pub(crate) tape: T,
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
    type Tape;
    type NoTape: Clone + PutTape<Self::Tape, Output = Self>;
    fn split_tape(self) -> (Self::NoTape, Self::Tape);
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T> SplitTape for Tensor<S, E, D, T> {
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

impl<S: Shape, E: Dtype, D: DeviceStorage, T> HasShape for Tensor<S, E, D, T> {
    type With<New: Shape> = Tensor<New, E, D, T>;
    type Shape = S;
    fn shape(&self) -> &Self::Shape {
        self.storage.shape()
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T> HasDtype for Tensor<S, E, D, T> {
    type Dtype = E;
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T> AllocGradOn<D> for Tensor<S, E, D, T> {
    fn try_alloc_grad(&self) -> Result<D::Storage<Self::Shape, Self::Dtype>, D::Err> {
        self.device.try_alloc_grad(&self.storage)
    }
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

pub type Tensor0D<D, Tape = NoneTape> = Tensor<Rank0, f32, D, Tape>;
pub type Tensor1D<const M: usize, D, Tape = NoneTape> = Tensor<Rank1<M>, f32, D, Tape>;
pub type Tensor2D<const M: usize, const N: usize, D, Tape = NoneTape> =
    Tensor<Rank2<M, N>, f32, D, Tape>;
pub type Tensor3D<const M: usize, const N: usize, const O: usize, D, Tape = NoneTape> =
    Tensor<Rank3<M, N, O>, f32, D, Tape>;
pub type Tensor4D<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    D,
    Tape = NoneTape,
> = Tensor<Rank4<M, N, O, P>, f32, D, Tape>;
pub type Tensor5D<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
    D,
    Tape = NoneTape,
> = Tensor<Rank5<M, N, O, P, Q>, f32, D, Tape>;
pub type Tensor6D<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
    const R: usize,
    D,
    Tape = NoneTape,
> = Tensor<Rank6<M, N, O, P, Q, R>, f32, D, Tape>;
