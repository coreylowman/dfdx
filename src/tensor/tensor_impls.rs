use rand::distributions::Distribution;

use super::storage_traits::{DeviceStorage, HasErr, TensorFromVec};
use super::{Cpu, OneFillStorage, SampleTensor, ZeroFillStorage};
use crate::{
    gradients::{Gradients, NoneTape, OwnedTape, Tape},
    shapes::*,
    unique_id::{HasUniqueId, UniqueId},
};

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

impl<S: Shape, E: Dtype, D: DeviceStorage, T> HasUniqueId for Tensor<S, E, D, T> {
    fn id(&self) -> &UniqueId {
        &self.id
    }
}

impl<S: Shape, E: Unit, D: DeviceStorage, T> HasErr for Tensor<S, E, D, T> {
    type Err = D::Err;
}

impl<S: Shape, E: Unit, D: DeviceStorage> Tensor<S, E, D, NoneTape> {
    /// Clone and put a [OwnedTape] into the tensor
    pub fn trace<F: Unit>(&self) -> Tensor<S, E, D, OwnedTape<F, D>> {
        self.clone().traced()
    }
    /// Put a [OwnedTape] into the tensor
    pub fn traced<F: Unit>(self) -> Tensor<S, E, D, OwnedTape<F, D>> {
        self.put_tape(Default::default())
    }
    /// Clone and put a [OwnedTape] into the tensor with an existing [Gradients]
    /// object. Enables gradient accumuluation!
    pub fn trace_with<F: Unit>(
        &self,
        gradients: Gradients<F, D>,
    ) -> Tensor<S, E, D, OwnedTape<F, D>> {
        self.clone().traced_with(gradients)
    }
    /// Put a [OwnedTape] into the tensor with an existing [Gradients] object.
    /// Enables gradient accumuluation!
    pub fn traced_with<F: Unit>(
        self,
        gradients: Gradients<F, D>,
    ) -> Tensor<S, E, D, OwnedTape<F, D>> {
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
    type Tape: Default;
    // The type of Self without the tape.
    type NoTape: Clone + PutTape<Self::Tape, Output = Self>;
    /// Splits tape off of self
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank1<5>, f32, _, OwnedTape<f32, _>> = dev.zeros().traced();
    /// let (a, tape): (Tensor<_, _, _, NoneTape>, OwnedTape<f32, _>) = a.split_tape();
    /// ```
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
                data: self.data,
                shape: self.shape,
                strides: self.strides,
                device: self.device,
                tape: NoneTape,
            },
            self.tape,
        )
    }

    fn with_empty_tape(&self) -> Self {
        Self {
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

/// Something that can be copied to another `Device` and can be used with the [OnDevice] type
/// alias.
///
/// Here's an example of how this can be implemented for a custom struct:
/// ```rust
/// use dfdx::{prelude::*, nn::modules::Linear};
///
/// struct MLP<D: Device<f32>> {
///     l1: Linear<5, 10, f32, D>,
///     a1: ReLU,
///     l2: Linear<10, 1, f32, D>,
/// }
///
/// // Need two device types to allow converting from one device to another
/// impl<D1: Device<f32>, D2: Device<f32>> ToDevice<D2> for MLP<D1> {
///     type Output = MLP<D2>;
///
///     fn to_device(&self, device: &D2) -> Self::Output {
///         MLP {
///             l1: self.l1.to_device(device),
///             a1: self.a1,
///             l2: self.l2.to_device(device),
///         }
///     }
/// }
/// ````
pub trait ToDevice<D> {
    type Output;
    fn to_device(&self, device: &D) -> Self::Output;
}

/// A type alias that yields the type of a module `M` as it would exist on device `D`. This can be
/// useful when creating sequential networks that need to be parameterized by a device.
pub type OnDevice<M, D> = <M as ToDevice<D>>::Output;

/// Equivalent to `OnDevice<M, Cuda>`
#[cfg(feature = "cuda")]
pub type OnCuda<M> = OnDevice<M, crate::prelude::Cuda>;

/// Equivalent to `OnDevice<M, Cpu>`
pub type OnCpu<M> = OnDevice<M, Cpu>;

impl<S: Shape, E: Dtype + Unit, T, D1: DeviceStorage, D2: TensorFromVec<E>> ToDevice<D2>
    for Tensor<S, E, D1, T>
{
    type Output = Tensor<S, E, D2, NoneTape>;

    fn to_device(&self, device: &D2) -> Self::Output {
        let buf = self.as_vec();
        device.tensor_from_vec(buf, *self.shape())
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
