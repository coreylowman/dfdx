use super::*;
use crate::gradients::Tape;

/// Changes the kind of tape inside a tensor.
pub trait PutTape<H: Tape> {
    type Output;
    /// Replaces whatever tape is in `self` with `tape`.
    fn put_tape(self, tape: H) -> Self::Output;

    /// Clones `self` and put's a brand new tape on it
    fn with_diff_tape(&self) -> Self::Output
    where
        Self: Clone,
    {
        self.clone().put_tape(Default::default())
    }
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* HIn: Tape, HOut: Tape> PutTape<HOut> for $typename<$($Vs, )* HIn>
{
    type Output = $typename<$($Vs, )* HOut>;
    fn put_tape(self, tape: HOut) -> Self::Output {
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
tensor_impl!(Tensor5D, [M, N, O, P, Q]);
tensor_impl!(Tensor6D, [M, N, O, P, Q, R]);
