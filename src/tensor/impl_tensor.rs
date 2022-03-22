use super::*;

pub trait Tensor: IsShapedArray + CanUpdateWithTape + HasUniqueId {
    type TapeHolder: TapeHolder;
    type NoTape: Tensor<TapeHolder = NoTape, Dimension = Self::Dimension>;
    type WithTape: Tensor<TapeHolder = WithTape, Dimension = Self::Dimension>;

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
