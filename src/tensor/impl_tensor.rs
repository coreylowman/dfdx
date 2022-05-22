use crate::prelude::*;

pub trait Tensor:
    HasArrayType
    + HasArrayData
    + HasDevice
    + CanUpdateWithGradients
    + HasUniqueId
    + IntoPhantom
    + CountElements<Dtype = <Self::Array as CountElements>::Dtype>
{
    type TapeHolder: TapeHolder;

    type NoTape: 'static
        + Tensor<TapeHolder = NoTape, Array = Self::Array, Device = Self::Device>
        // NOTE: we only want to be able to create NoTape tensors
        + TensorCreator
        // NOTE: Adding this restriction means we can put the tape from Self into the Self::NoTape
        + CanPutTapeHolder<Self::TapeHolder, Output = Self>;

    type WithTape: 'static
        + Tensor<TapeHolder = WithTape, Array = Self::Array, Device = Self::Device>;

    fn split_tape_holder(self) -> (Self::NoTape, Self::TapeHolder);
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> Tensor for $typename<$($Vs, )* H> {
    type TapeHolder = H;
    type NoTape = $typename<$($Vs, )* NoTape>;
    type WithTape = $typename<$($Vs, )* WithTape>;

    fn split_tape_holder(self) -> (Self::NoTape, Self::TapeHolder) {
        (
            Self::NoTape { id: self.id, data: self.data, tape: NoTape::default() },
            self.tape,
        )
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
