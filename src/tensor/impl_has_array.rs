use super::*;
use crate::prelude::*;

pub trait HasArrayData: HasArrayType {
    fn data(&self) -> &Self::Array;
    fn mut_data(&mut self) -> &mut Self::Array;
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*], $arr:ty) => {
impl<$(const $Vs: usize, )* H> HasArrayType for $typename<$($Vs, )* H>  {
    type Array = $arr;
}

impl<$(const $Vs: usize, )* H> HasArrayData for $typename<$($Vs, )* H> {
    fn data(&self) -> &Self::Array { &self.data }
    fn mut_data(&mut self) -> &mut Self::Array { &mut self.data }
}

impl<$(const $Vs: usize, )* H> CountElements for $typename<$($Vs, )* H> {
    const NUM_ELEMENTS: usize = <Self as HasArrayType>::Array::NUM_ELEMENTS;
    type Dtype = <<Self as HasArrayType>::Array as CountElements>::Dtype;
}
    };
}

tensor_impl!(Tensor0D, [], f32);
tensor_impl!(Tensor1D, [M], [f32; M]);
tensor_impl!(Tensor2D, [M, N], [[f32; N]; M]);
tensor_impl!(Tensor3D, [M, N, O], [[[f32; O]; N]; M]);
tensor_impl!(Tensor4D, [M, N, O, P], [[[[f32; P]; O]; N]; M]);
