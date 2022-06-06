use crate::prelude::*;

/// The main tensor trait. A tensor consists of mainly 1. an array, 2. a device, 3. a unique id.
pub trait Tensor:
    HasArrayType + HasArrayData + HasDevice + CanUpdateWithGradients + HasUniqueId + IntoPhantom
{
    type Tape: Tape;

    type NoTape: 'static
        + Tensor<Array = Self::Array, Dtype = Self::Dtype, Tape = NoTape, NoTape = Self::NoTape>
        // NOTE: we only want to be able to create NoTape tensors
        + TensorCreator
        // NOTE: Adding this restriction means we can put the tape from Self into the Self::NoTape
        + PutTape<Self::Tape, Output = Self>
        + Clone;

    type OwnsTape: 'static
        + Tensor<Array = Self::Array, Dtype = Self::Dtype, Tape = OwnsTape, OwnsTape = Self::OwnsTape>;

    type LastDimReduced: Tensor<
        Tape = Self::Tape,
        Dtype = Self::Dtype,
        Array = <Self::Device as ReduceLastDim<Self::Array>>::Reduced,
    >;

    /// Removes whatever Tape this tensor has and returns itself without a tape.
    fn split_tape(self) -> (Self::NoTape, Self::Tape);

    /// Clones the data & [UniqueId] of this tensor and returns something with [NoTape].
    fn duplicate(&self) -> Self::NoTape;
}

macro_rules! tensor_impl {
    ($struct:ident, [$($Vs:tt),*], $reduced:ident, [$($Rs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> Tensor for $struct<$($Vs, )* H> {
    type Tape = H;
    type NoTape = $struct<$($Vs, )* NoTape>;
    type OwnsTape = $struct<$($Vs, )* OwnsTape>;

    type LastDimReduced = $reduced<$($Rs, )* H>;

    fn split_tape(self) -> (Self::NoTape, Self::Tape) {
        (
            Self::NoTape { id: self.id, data: self.data, tape: NoTape::default() },
            self.tape,
        )
    }

    fn duplicate(&self) -> Self::NoTape {
        Self::NoTape {
            id: self.id,
            data: self.data.clone(),
            tape: NoTape::default(),
        }
    }
}

impl<$(const $Vs: usize, )* H: Clone> Clone for $struct<$($Vs, )* H> {
    /// Clones the underlying data and tape. **Creates a new [UniqueId].**
    fn clone(&self) -> Self {
        Self {
            id: unique_id(),
            data: self.data.clone(),
            tape: self.tape.clone(),
        }
    }
}
    };
}

tensor_impl!(Tensor0D, [], Tensor0D, []);
tensor_impl!(Tensor1D, [M], Tensor0D, []);
tensor_impl!(Tensor2D, [M, N], Tensor1D, [M]);
tensor_impl!(Tensor3D, [M, N, O], Tensor2D, [M, N]);
tensor_impl!(Tensor4D, [M, N, O, P], Tensor3D, [M, N, O]);
