use rand::distributions::Distribution;

use super::*;
use crate::shapes::*;

use std::sync::Arc;

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
/// type B = Tensor<Rank2<2, 3>, bool, Cpu>;
///
/// // A 3d tensor with usize elements, stored on the Cpu, without any tape
/// type C = Tensor<Rank3<4, 2, 3>, usize, Cpu, NoneTape>;
/// ```
#[derive(Debug, Clone)]
pub struct Tensor<S: Shape, E: Unit, D: DeviceStorage, T = NoneTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: Arc<D::Vec<E>>,
    pub(crate) shape: S,
    pub(crate) strides: S::Concrete,
    pub(crate) device: D,
    pub(crate) tape: T,
}

impl<S: Shape, E: Unit, D: DeviceStorage, T> HasShape for Tensor<S, E, D, T> {
    type WithShape<New: Shape> = Tensor<New, E, D, T>;
    type Shape = S;
    fn shape(&self) -> &Self::Shape {
        &self.shape
    }
}

impl<S: Shape, E: Unit, D: DeviceStorage, T> HasUnitType for Tensor<S, E, D, T> {
    type Unit = E;
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T> HasDtype for Tensor<S, E, D, T> {
    type Dtype = E;
}

impl<S: Shape, E: Unit, D: DeviceStorage, T> HasErr for Tensor<S, E, D, T> {
    type Err = D::Err;
}

/// Something that can trace gradients
pub trait Trace<E: Unit, D: DeviceStorage>: Clone {
    type Traced;
    /// Start tracking gradients, clones self. The gradients will never free
    /// temporary gradients - See [Gradients::leaky()] for more info.
    ///
    /// Prefer to use [Tensor::trace()] with gradients allocated
    /// with [crate::nn::ZeroGrads::alloc_grads()].
    fn leaky_trace(&self) -> Self::Traced {
        self.clone().leaky_traced()
    }
    /// Start tracking gradients. The gradients will never free
    /// temporary gradients - See [Gradients::leaky()] for more info.
    ///
    /// Prefer to use [Tensor::traced()] with gradients allocated
    /// with [crate::nn::ZeroGrads::alloc_grads()].
    fn leaky_traced(self) -> Self::Traced;

    /// Accumulates gradients into `gradients`, clones self. Use [crate::nn::ZeroGrads::alloc_grads()]
    /// to create gradients.
    fn trace(&self, gradients: Gradients<E, D>) -> Self::Traced {
        self.clone().traced(gradients)
    }
    /// Accumulates gradients into `gradients`. Use [crate::nn::ZeroGrads::alloc_grads()]
    /// to create gradients.
    fn traced(self, gradients: Gradients<E, D>) -> Self::Traced;
}

impl<S: Shape, E: Unit, F: Unit, D: DeviceStorage> Trace<E, D> for Tensor<S, F, D, NoneTape> {
    type Traced = Tensor<S, F, D, OwnedTape<E, D>>;
    fn leaky_traced(self) -> Self::Traced {
        self.put_tape(Default::default())
    }
    fn traced(self, gradients: Gradients<E, D>) -> Self::Traced {
        self.put_tape(OwnedTape {
            gradients,
            operations: std::vec::Vec::new(),
        })
    }
}

impl<S: Shape, E: Unit, D: DeviceStorage, T> Tensor<S, E, D, T> {
    /// Clone and insert a new tape of type `New` into the tensor
    pub fn retaped<New: Tape<E, D>>(&self) -> Tensor<S, E, D, New> {
        Tensor {
            id: self.id,
            data: self.data.clone(),
            shape: self.shape,
            strides: self.strides,
            device: self.device.clone(),
            tape: Default::default(),
        }
    }
}

/// Put a tape of type `T` into the tensor
pub trait PutTape<T> {
    type Output;
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32, _, NoneTape> = dev.zeros();
    /// let a: Tensor<Rank2<2, 3>, f32, _, OwnedTape<f32, Cpu>> = a.put_tape(Default::default());
    /// ```
    fn put_tape(self, tape: T) -> Self::Output;
}

impl<S: Shape, E: Unit, D: DeviceStorage, T> PutTape<T> for Tensor<S, E, D> {
    type Output = Tensor<S, E, D, T>;
    fn put_tape(self, tape: T) -> Self::Output {
        Tensor {
            id: self.id,
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            device: self.device,
            tape,
        }
    }
}

/// Remove the tape from a tensor
pub trait SplitTape {
    /// The type of tape the tensor has now
    type Tape;
    // The type of Self without the tape.
    type NoTape: Clone + PutTape<Self::Tape, Output = Self>;
    /// Splits tape off of self
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// # let grads = Gradients::leaky();
    /// let a: Tensor<Rank1<5>, f32, _, OwnedTape<f32, _>> = dev.zeros().traced(grads);
    /// let (a, tape): (Tensor<_, _, _, NoneTape>, OwnedTape<f32, _>) = a.split_tape();
    /// ```
    fn split_tape(self) -> (Self::NoTape, Self::Tape);
}

impl<S: Shape, E: Unit, D: DeviceStorage, T> SplitTape for Tensor<S, E, D, T> {
    type Tape = T;
    type NoTape = Tensor<S, E, D>;
    fn split_tape(self) -> (Self::NoTape, Self::Tape) {
        (
            Tensor {
                id: self.id,
                data: self.data,
                shape: self.shape,
                strides: self.strides,
                device: self.device,
                tape: NoneTape,
            },
            self.tape,
        )
    }
}

/// Clones self and inserts a new empty tape into the clone
pub trait WithEmptyTape {
    /// Clones self and inserts a new empty tape into the clone
    fn with_empty_tape(&self) -> Self;
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T: Default> WithEmptyTape for Tensor<S, E, D, T> {
    fn with_empty_tape(&self) -> Self {
        Tensor {
            id: self.id,
            data: self.data.clone(),
            shape: self.shape,
            strides: self.strides,
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
        self.device
            .try_fill_with_zeros(Arc::make_mut(&mut self.data))
    }
}

impl<S: Shape, E: Dtype, D: OneFillStorage<E>, T> Tensor<S, E, D, T> {
    /// Fills the tensor with ones
    pub fn fill_with_ones(&mut self) {
        self.try_fill_with_ones().unwrap()
    }
    /// Fallible version of [Tensor::fill_with_ones]
    pub fn try_fill_with_ones(&mut self) -> Result<(), D::Err> {
        self.device
            .try_fill_with_ones(Arc::make_mut(&mut self.data))
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
        self.device
            .try_fill_with_distr(Arc::make_mut(&mut self.data), distr)
    }
}

pub type Tensor0D<Tape = NoneTape> = Tensor<Rank0, f32, Cpu, Tape>;
pub type Tensor1D<const M: usize, Tape = NoneTape> = Tensor<Rank1<M>, f32, Cpu, Tape>;
pub type Tensor2D<const M: usize, const N: usize, Tape = NoneTape> =
    Tensor<Rank2<M, N>, f32, Cpu, Tape>;
pub type Tensor3D<const M: usize, const N: usize, const O: usize, Tape = NoneTape> =
    Tensor<Rank3<M, N, O>, f32, Cpu, Tape>;
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
