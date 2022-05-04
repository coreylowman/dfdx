use super::*;
use crate::gradients::{HasNdArray, HasUniqueId};

pub trait Tensor: HasNdArray + CanUpdateWithTape + HasUniqueId + IntoPhantom {
    type TapeHolder: TapeHolder;

    type NoTape: 'static
        + Tensor<TapeHolder = NoTape, ArrayType = Self::ArrayType>
        + TensorCreator
        // NOTE: Adding this restriction means we can put the tape from Self into the Self::NoTape
        + HasTapeHolder<Self::TapeHolder, Output = Self>
        + IntoPhantom;

    type WithTape: 'static
        + Tensor<TapeHolder = WithTape, ArrayType = Self::ArrayType>
        + IntoPhantom;

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
