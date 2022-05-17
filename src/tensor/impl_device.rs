use super::*;
use crate::prelude::Device;

pub trait HasDevice: HasNdArray {
    type Device: Device<Self::ArrayType>;
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H, D: Device<Self::ArrayType>> HasDevice for $typename<$($Vs, )* H, D> {
    type Device = D;
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
