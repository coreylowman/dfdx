use super::*;

pub trait HasTapeHolder<H: TapeHolder> {
    type Output;
    fn with_tape_holder(self, tape_holder: H) -> Self::Output;
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* HIn, HOut> HasTapeHolder<HOut> for $typename<$($Vs, )* HIn>
where
    HIn: TapeHolder,
    HOut: TapeHolder,
{
    type Output = $typename<$($Vs, )* HOut>;
    fn with_tape_holder(self, tape: HOut) -> Self::Output {
        Self::Output { id: self.id, data: self.data, tape }
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
