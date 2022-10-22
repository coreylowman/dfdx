use super::*;
use crate::gradients::Tape;

pub trait PutTape<H: Tape> {
    type Output;
    fn put_tape(self, tape: H) -> Self::Output;
    fn retaped(&self) -> Self::Output;
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* HIn: Tape, HOut: Tape> PutTape<HOut> for $typename<$($Vs, )* HIn>
{
    type Output = $typename<$($Vs, )* HOut>;
    fn put_tape(self, tape: HOut) -> Self::Output {
        Self::Output { id: self.id, data: self.data, tape }
    }
    fn retaped(&self) -> Self::Output {
        Self::Output { id: self.id, data: self.data.clone(), tape: HOut::default() }
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
