use crate::prelude::*;

/// The main tensor trait. A tensor consists of mainly 1. an array, 2. a device, 3. a unique id.
pub trait Tensor:
    HasArrayType + HasArrayData + HasDevice + CanUpdateWithGradients + HasUniqueId + IntoPhantom
{
    type TapeHolder: TapeHolder;

    type NoTape: 'static
        + Tensor<Array = Self::Array, Dtype = Self::Dtype>
        // NOTE: we only want to be able to create NoTape tensors
        + TensorCreator
        // NOTE: Adding this restriction means we can put the tape from Self into the Self::NoTape
        + CanPutTapeHolder<Self::TapeHolder, Output = Self>;

    type OwnsTape: 'static + Tensor<Array = Self::Array, Dtype = Self::Dtype, TapeHolder = OwnsTape>;

    type LastDimReduced: Tensor<
        TapeHolder = Self::TapeHolder,
        Array = <Self::Device as ReduceLastDim<Self::Array>>::Reduced,
        Dtype = Self::Dtype,
    >;

    /// Removes whatever TapeHolder this tensor has and returns itself without a tape.
    fn split_tape_holder(self) -> (Self::NoTape, Self::TapeHolder);

    /// Clones the data & [UniqueId] of this tensor and returns something with [NoTape].
    fn duplicate(&self) -> Self::NoTape;
}

macro_rules! tensor_impl {
    ($struct:ident, [$($Vs:tt),*], $reduced:ident, [$($Rs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> Tensor for $struct<$($Vs, )* H> {
    type TapeHolder = H;
    type NoTape = $struct<$($Vs, )* NoTape>;
    type OwnsTape = $struct<$($Vs, )* OwnsTape>;

    type LastDimReduced = $reduced<$($Rs, )* H>;

    fn split_tape_holder(self) -> (Self::NoTape, Self::TapeHolder) {
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
    };
}

tensor_impl!(Tensor0D, [], Tensor0D, []);
tensor_impl!(Tensor1D, [M], Tensor0D, []);
tensor_impl!(Tensor2D, [M, N], Tensor1D, [M]);
tensor_impl!(Tensor3D, [M, N, O], Tensor2D, [M, N]);
tensor_impl!(Tensor4D, [M, N, O, P], Tensor3D, [M, N, O]);
