use super::*;
use crate::gradients::NoTape;

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> Default for $typename<$($Vs, )* NoTape> {
    /// Returns a tensor with all elements equal to 0
    fn default() -> Self {
        Self::zeros()
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
