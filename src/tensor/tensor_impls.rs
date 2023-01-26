use rand::distributions::Distribution;

use super::storage_traits::{DeviceStorage, HasErr};
use super::{AutoDevice, OneFillStorage, SampleTensor, ZeroFillStorage};
use crate::{
    gradients::{NoneTape, OwnedTape, Tape},
    shapes::*,
    unique_id::{HasUniqueId, UniqueId},
};

/// The single tensor struct that stores nd arrays and tapes.
///
/// See module level documentation on how to create and use tensors.
///
/// Generics:
/// 1. [Shape] - the shape of the underlying nd array
/// 2. [Dtype] - the type of the datas stored in the array
/// 3. [DeviceStorage] - the device the array is stored on
/// 4. [Tape] - the tape the tensor has
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// // A 1d tensor with 1000 f32 elements, stored on the Cpu
/// type A = Tensor<Rank1<1000>, f32, Cpu>;
///
/// // A 2d tensor with bool elements, stored on the Cpu
/// type B = Tensor<Rank2<2, 3>, bool>;
///
/// // A 3d tensor with usize elements, stored on the Cpu, without any tape
/// type C = Tensor<Rank3<4, 2, 3>, usize, Cpu, NoneTape>;
/// ```
#[derive(Debug, Clone)]
pub struct Tensor<S: Shape, E: Unit = f32, D: DeviceStorage = AutoDevice, T = NoneTape> {
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

impl<S: Shape, E: Unit, D: DeviceStorage, T> HasUnitType for Tensor<S, E, D, T> {
    type Unit = E;
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
    /// Clone and put a [OwnedTape] into the tensor
    pub fn trace(&self) -> Tensor<S, E, D, OwnedTape<D>> {
        self.clone().traced()
    }
    /// Put a [OwnedTape] into the tensor
    pub fn traced(self) -> Tensor<S, E, D, OwnedTape<D>> {
        self.put_tape(Default::default())
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T: Tape<D>> Tensor<S, E, D, T> {
    /// Clone and insert a new tape of type `New` into the tensor
    pub fn retaped<New: Tape<D>>(&self) -> Tensor<S, E, D, New> {
        Tensor {
            id: self.id,
            storage: self.storage.clone(),
            device: self.device.clone(),
            tape: Default::default(),
        }
    }
}

/// Put a tape of type `T` into the tensor
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a: Tensor<Rank2<2, 3>> = dev.zeros();
/// let a: Tensor<Rank2<2, 3>, f32, _, OwnedTape<Cpu>> = a.put_tape(Default::default());
/// ```
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

/// Remove the tape from a tensor
///
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a: Tensor<Rank1<5>, f32, _, OwnedTape<Cpu>> = dev.zeros().traced();
/// let (a, tape): (Tensor<Rank1<5>, f32>, OwnedTape<Cpu>) = a.split_tape();
pub trait SplitTape {
    /// The type of tape the tensor has now
    type Tape: Default;
    // The type of Self without the tape.
    type NoTape: Clone + PutTape<Self::Tape, Output = Self>;
    /// Splits tape off of self
    fn split_tape(self) -> (Self::NoTape, Self::Tape);
    /// Clones self and inserts a new empty tape into the clone
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

impl<S: Shape, E: Dtype, D: ZeroFillStorage<E>, T> Tensor<S, E, D, T> {
    /// Fills the tensor with zeros
    pub fn fill_with_zeros(&mut self) {
        self.try_fill_with_zeros().unwrap()
    }
    /// Fallible version of [Tensor::fill_with_zeros]
    pub fn try_fill_with_zeros(&mut self) -> Result<(), D::Err> {
        self.device.try_fill_with_zeros(&mut self.storage)
    }
}

impl<S: Shape, E: Dtype, D: OneFillStorage<E>, T> Tensor<S, E, D, T> {
    /// Fills the tensor with ones
    pub fn fill_with_ones(&mut self) {
        self.try_fill_with_ones().unwrap()
    }
    /// Fallible version of [Tensor::fill_with_ones]
    pub fn try_fill_with_ones(&mut self) -> Result<(), D::Err> {
        self.device.try_fill_with_ones(&mut self.storage)
    }
}

impl<S: Shape, E: Unit, D: SampleTensor<E>, T> Tensor<S, E, D, T> {
    /// Fills the tensor with random data from the distribution
    pub fn fill_with_distr<Distr: Distribution<E>>(&mut self, distr: Distr) {
        self.try_fill_with_distr(distr).unwrap()
    }

    /// Fallible version of [Tensor::fill_with_distr]
    pub fn try_fill_with_distr<Distr: Distribution<E>>(
        &mut self,
        distr: Distr,
    ) -> Result<(), D::Err> {
        self.device.try_fill_with_distr(&mut self.storage, distr)
    }
}

pub type Tensor0D<Tape = NoneTape> = Tensor<Rank0, f32, AutoDevice, Tape>;
pub type Tensor1D<const M: usize, Tape = NoneTape> = Tensor<Rank1<M>, f32, AutoDevice, Tape>;
pub type Tensor2D<const M: usize, const N: usize, Tape = NoneTape> =
    Tensor<Rank2<M, N>, f32, AutoDevice, Tape>;
pub type Tensor3D<const M: usize, const N: usize, const O: usize, Tape = NoneTape> =
    Tensor<Rank3<M, N, O>, f32, AutoDevice, Tape>;
pub type Tensor4D<const M: usize, const N: usize, const O: usize, const P: usize, Tape = NoneTape> =
    Tensor<Rank4<M, N, O, P>, f32, AutoDevice, Tape>;
pub type Tensor5D<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
    Tape = NoneTape,
> = Tensor<Rank5<M, N, O, P, Q>, f32, AutoDevice, Tape>;
pub type Tensor6D<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
    const R: usize,
    Tape = NoneTape,
> = Tensor<Rank6<M, N, O, P, Q, R>, f32, AutoDevice, Tape>;
