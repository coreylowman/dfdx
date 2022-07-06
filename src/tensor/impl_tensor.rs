use crate::prelude::*;

/// The main tensor trait. A tensor consists of mainly 1. an array, 2. a device, 3. a unique id.
pub trait Tensor:
    HasArrayType + HasArrayData + HasDevice + CanUpdateWithGradients + HasUniqueId + IntoPhantom
{
    /// The [Tape] this tensor owns.
    type Tape: Tape;

    /// This tensor but with [NoTape].
    type NoTape: 'static
        + Tensor<Array = Self::Array, Dtype = Self::Dtype, Tape = NoneTape, NoTape = Self::NoTape>
        // NOTE: we only want to be able to create NoTape tensors
        + TensorCreator
        // NOTE: Adding this restriction means we can put the tape from Self into the Self::NoTape
        + PutTape<Self::Tape, Output = Self>
        + Clone;

    /// This tensor but with [OwnedTape]
    type OwnedTape: 'static
        + Tensor<
            Array = Self::Array,
            Dtype = Self::Dtype,
            Tape = OwnedTape,
            OwnedTape = Self::OwnedTape,
        >;

    /// This tensor but with it's last dimension reduced to 1. See [ReduceLastDim].
    type LastDimReduced: Tensor<
        Tape = Self::Tape,
        Dtype = Self::Dtype,
        Array = <Self::Device as ReduceLastDim<Self::Array>>::Reduced,
    >;

    /// Indices used for [gather_last_dim()] that can reduce this tensor to it's [Tensor::LastDimReduced].
    type ReducingIndices: CountElements<Dtype = usize>;

    /// Removes whatever Tape this tensor has and returns itself without a tape.
    fn split_tape(self) -> (Self::NoTape, Self::Tape);

    /// Clones the data & [UniqueId] of this tensor and returns something with [NoTape].
    fn duplicate(&self) -> Self::NoTape;
}

macro_rules! tensor_impl {
    ($struct:ident, [$($Vs:tt),*], $reduced:ident, [$($Rs:tt),*], $ix:ty) => {
impl<$(const $Vs: usize, )* H: Tape> Tensor for $struct<$($Vs, )* H> {
    type Tape = H;
    type NoTape = $struct<$($Vs, )* NoneTape>;
    type OwnedTape = $struct<$($Vs, )* OwnedTape>;

    type LastDimReduced = $reduced<$($Rs, )* H>;
    type ReducingIndices = $ix;

    fn split_tape(self) -> (Self::NoTape, Self::Tape) {
        (
            Self::NoTape { id: self.id, data: self.data, tape: Default::default() },
            self.tape,
        )
    }

    fn duplicate(&self) -> Self::NoTape {
        Self::NoTape {
            id: self.id,
            data: self.data.clone(),
            tape: Default::default(),
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

tensor_impl!(Tensor0D, [], Tensor0D, [], usize);
tensor_impl!(Tensor1D, [M], Tensor0D, [], usize);
tensor_impl!(Tensor2D, [M, N], Tensor1D, [M], [usize; M]);
tensor_impl!(Tensor3D, [M, N, O], Tensor2D, [M, N], [[usize; N]; M]);
tensor_impl!(
    Tensor4D,
    [M, N, O, P],
    Tensor3D,
    [M, N, O],
    [[[usize; O]; N]; M]
);
