use super::*;
use crate::prelude::*;

/// Something that has [HasArrayType], and also can return a reference to or mutate `Self::Array`.
pub trait HasArrayData: HasArrayType {
    fn data(&self) -> &Self::Array;
    fn mut_data(&mut self) -> &mut Self::Array;
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*], $arr:ty) => {
impl<$(const $Vs: usize, )* H> HasArrayType for $typename<$($Vs, )* H>  {
    type Dtype = f32;
    type Array = $arr;
}

impl<$(const $Vs: usize, )* H> HasArrayData for $typename<$($Vs, )* H> {
    /// Returns a reference to the underlying array.
    fn data(&self) -> &Self::Array { self.data.as_ref() }

    /// Returns a mutable reference to the underlying array.
    fn mut_data(&mut self) -> &mut Self::Array { std::rc::Rc::make_mut(&mut self.data) }
}
    };
}

tensor_impl!(Tensor0D, [], f32);
tensor_impl!(Tensor1D, [M], [f32; M]);
tensor_impl!(Tensor2D, [M, N], [[f32; N]; M]);
tensor_impl!(Tensor3D, [M, N, O], [[[f32; O]; N]; M]);
tensor_impl!(Tensor4D, [M, N, O, P], [[[[f32; P]; O]; N]; M]);
