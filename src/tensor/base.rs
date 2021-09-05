use crate::{
    gradients::*,
    traits::{Params, RandomInit, ShapedArray, Tensor},
};
use ndarray::prelude::*;
use ndarray_rand::rand::Rng;

#[derive(Debug)]
pub struct Tensor0D {
    pub(super) data: Array0<f32>,
    pub(super) grad: Option<Grad>,
}

impl Default for Tensor0D {
    fn default() -> Self {
        Self {
            data: Array::zeros(Self::SHAPE),
            grad: None,
        }
    }
}

impl ShapedArray for Tensor0D {
    type Dimension = Ix0;
    type Shape = ();
    const SHAPE: Self::Shape = ();

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }

    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

impl Tensor for Tensor0D {
    fn grad(&self) -> &Option<Grad> {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut Option<Grad> {
        &mut self.grad
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
            data: Array::zeros(Self::SHAPE),
            grad: None,
        }
    }
}

impl<const N: usize> ShapedArray for Tensor1D<N> {
    type Dimension = Ix1;
    type Shape = (usize,);
    const SHAPE: Self::Shape = (N,);

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }

    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

impl<const N: usize> Tensor for Tensor1D<N> {
    fn grad(&self) -> &Option<Grad> {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut Option<Grad> {
        &mut self.grad
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
            data: Array::zeros(Self::SHAPE),
            grad: Default::default(),
        }
    }
}

impl<const M: usize, const N: usize> ShapedArray for Tensor2D<M, N> {
    type Dimension = Ix2;
    type Shape = (usize, usize);
    const SHAPE: Self::Shape = (M, N);
    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }

    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

impl<const M: usize, const N: usize> Tensor for Tensor2D<M, N> {
    fn grad(&self) -> &Option<Grad> {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut Option<Grad> {
        &mut self.grad
    }
}

impl<T: Tensor> RandomInit for T {
    fn randomize<R: Rng>(&mut self, rng: &mut R) {
        self.mut_data().map_inplace(|f| *f = rng.gen())
    }
}

impl<T: Tensor> Params for T {
    fn register(&mut self, tape: &mut GradientTape) {
        if self.grad().is_none() {
            *self.mut_grad() = Some(Grad::new(tape.store_gradient(Self::SHAPE)));
        }
    }

    fn update(&mut self, tape: &GradientTape) {
        let grad = self.mut_grad().take().unwrap();
        *self.mut_data() -= &tape[grad.gradient_ref];
    }
}
