use crate::{gradients::*, traits::Tensor};
use ndarray::prelude::*;

#[derive(Debug)]
pub struct Tensor0D {
    pub(super) data: Array0<f32>,
    pub(super) grad: Option<Grad>,
}

impl Default for Tensor0D {
    fn default() -> Self {
        Self {
            data: Array0::zeros(()),
            grad: None,
        }
    }
}

impl Tensor for Tensor0D {
    type Dimension = Ix0;
    type Shape = ();
    const SHAPE: Self::Shape = ();

    fn grad(&self) -> &Option<Grad> {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut Option<Grad> {
        &mut self.grad
    }

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }

    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

#[derive(Debug)]
pub struct Tensor1D<const N: usize> {
    pub(super) data: Array1<f32>,
    pub(super) grad: Option<Grad>,
}

impl<const N: usize> Default for Tensor1D<N> {
    fn default() -> Self {
        Self {
            data: Array1::zeros((N,)),
            grad: None,
        }
    }
}

impl<const N: usize> Tensor for Tensor1D<N> {
    type Dimension = Ix1;
    type Shape = (usize,);
    const SHAPE: Self::Shape = (N,);

    fn grad(&self) -> &Option<Grad> {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut Option<Grad> {
        &mut self.grad
    }

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }

    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

#[derive(Debug)]
pub struct Tensor2D<const M: usize, const N: usize> {
    pub(super) data: Array2<f32>,
    pub(super) grad: Option<Grad>,
}

impl<const M: usize, const N: usize> Default for Tensor2D<M, N> {
    fn default() -> Self {
        Self {
            data: Array2::zeros((M, N)),
            grad: Default::default(),
        }
    }
}

impl<const M: usize, const N: usize> Tensor for Tensor2D<M, N> {
    type Dimension = Ix2;
    type Shape = (usize, usize);
    const SHAPE: Self::Shape = (M, N);

    fn grad(&self) -> &Option<Grad> {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut Option<Grad> {
        &mut self.grad
    }

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }

    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}
