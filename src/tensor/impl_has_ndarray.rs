use super::*;
use crate::gradients::IsNdArray;

pub trait HasNdArray: IsNdArray {
    fn data(&self) -> &Self::ArrayType;
    fn mut_data(&mut self) -> &mut Self::ArrayType;
}

macro_rules! ndarray_impl {
    ($typename:ident, [$($Vs:tt),*], $arr:ty) => {
impl<$(const $Vs: usize, )* H> IsNdArray for $typename<$($Vs, )* H> {
    type ArrayType = $arr;
}

impl<$(const $Vs: usize, )* H> HasNdArray for $typename<$($Vs, )* H> {
    fn data(&self) -> &Self::ArrayType { &self.data }
    fn mut_data(&mut self) -> &mut Self::ArrayType { &mut self.data }
}

    };
}

ndarray_impl!(Tensor0D, [], f32);
ndarray_impl!(Tensor1D, [M], [f32; M]);
ndarray_impl!(Tensor2D, [M, N], [[f32; N]; M]);
ndarray_impl!(Tensor3D, [M, N, O], [[[f32; O]; N]; M]);
ndarray_impl!(Tensor4D, [M, N, O, P], [[[[f32; P]; O]; N]; M]);
