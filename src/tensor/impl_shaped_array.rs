use super::structs::*;
use super::traits::IsShapedArray;
use ndarray::{Array, Ix0, Ix1, Ix2, Ix3, Ix4};

impl<Tape> IsShapedArray for Tensor0D<Tape> {
    type Dimension = Ix0;
    type Shape = ();
    const SHAPE: Self::Shape = ();
    const NUM_ELEMENTS: usize = 1;

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

impl<const N: usize, Tape> IsShapedArray for Tensor1D<N, Tape> {
    type Dimension = Ix1;
    type Shape = (usize,);
    const SHAPE: Self::Shape = (N,);
    const NUM_ELEMENTS: usize = N;

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

impl<const M: usize, const N: usize, Tape> IsShapedArray for Tensor2D<M, N, Tape> {
    type Dimension = Ix2;
    type Shape = (usize, usize);
    const SHAPE: Self::Shape = (M, N);
    const NUM_ELEMENTS: usize = M * N;

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

impl<const M: usize, const N: usize, const O: usize, Tape> IsShapedArray
    for Tensor3D<M, N, O, Tape>
{
    type Dimension = Ix3;
    type Shape = (usize, usize, usize);
    const SHAPE: Self::Shape = (M, N, O);
    const NUM_ELEMENTS: usize = M * N * O;

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}
impl<const M: usize, const N: usize, const O: usize, const P: usize, Tape> IsShapedArray
    for Tensor4D<M, N, O, P, Tape>
{
    type Dimension = Ix4;
    type Shape = (usize, usize, usize, usize);
    const SHAPE: Self::Shape = (M, N, O, P);
    const NUM_ELEMENTS: usize = M * N * O * P;

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}
