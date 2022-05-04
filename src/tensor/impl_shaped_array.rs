use super::*;
use crate::gradients::HasNdArray;
// use ndarray::{Array, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, ShapeBuilder};

// pub trait IsShapedArray: HasNdArray {
//     type Dimension: Dimension;
//     type Shape: ShapeBuilder<Dim = Self::Dimension>;
//     const SHAPE: Self::Shape;
//     const SHAPE_SLICE: &'static [usize];
//     const NUM_ELEMENTS: usize;

//     fn data(&self) -> &Self::ArrayType;
//     fn mut_data(&mut self) -> &mut Self::ArrayType;

//     fn shape(&self) -> Self::Shape {
//         Self::SHAPE
//     }
// }

// macro_rules! tensor_impl {
//     ($typename:ident, $dim: ty, [$($Vs:tt),*], $shape:ty) => {
// impl<$(const $Vs: usize, )* H> IsShapedArray for $typename<$($Vs, )* H> {
//     type Dimension = $dim;
//     type Shape = $shape;
//     const SHAPE: Self::Shape = ($($Vs, )*);
//     const SHAPE_SLICE: &'static [usize] = &[$($Vs, )*];
//     const NUM_ELEMENTS: usize = $($Vs * )* 1;
//     fn data(&self) -> &Array<f32, Self::Dimension> { &self.data }
//     fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> { &mut self.data }
// }
//     };
// }

// tensor_impl!(Tensor0D, Ix0, [], ());
// tensor_impl!(Tensor1D, Ix1, [M], (usize,));
// tensor_impl!(Tensor2D, Ix2, [M, N], (usize, usize));
// tensor_impl!(Tensor3D, Ix3, [M, N, O], (usize, usize, usize));
// tensor_impl!(Tensor4D, Ix4, [M, N, O, P], (usize, usize, usize, usize));

macro_rules! ndarray_impl {
    ($typename:ident, [$($Vs:tt),*], $arr:ty, $zeros:expr) => {
impl<$(const $Vs: usize, )* H> HasNdArray for $typename<$($Vs, )* H> {
    type ArrayType = $arr;
    fn data(&self) -> &Self::ArrayType { &self.data }
    fn mut_data(&mut self) -> &mut Self::ArrayType { &mut self.data }
}
    };
}

ndarray_impl!(Tensor0D, [], f32, 0.0);
ndarray_impl!(Tensor1D, [M], [f32; M], [0.0; M]);
ndarray_impl!(Tensor2D, [M, N], [[f32; N]; M], [[0.0; N]; M]);
ndarray_impl!(Tensor3D, [M, N, O], [[[f32; O]; N]; M], [[[0.0; O]; N]; M]);
ndarray_impl!(
    Tensor4D,
    [M, N, O, P],
    [[[[f32; P]; O]; N]; M],
    [[[[0.0; P]; O]; N]; M]
);
