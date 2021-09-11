use crate::gradients::{Grad, GradientRef, GradientTape};
use ndarray::{Array, Dimension, ShapeBuilder};
use ndarray_rand::rand::prelude::*;
use std::ops::DerefMut;

pub trait RandomInit {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D);
}

pub trait Params {
    fn register(&mut self, tape: &mut GradientTape);
    fn update(&mut self, tape: &GradientTape);
}

pub trait ShapedArray {
    type Dimension: Dimension;
    type Shape: ShapeBuilder<Dim = Self::Dimension>;
    const SHAPE: Self::Shape;
    const NUM_ELEMENTS: usize;

    fn data(&self) -> &Array<f32, Self::Dimension>;
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension>;
}

pub trait Activations {
    fn relu(&mut self) -> Self;
    fn sin(&mut self) -> Self;
    fn cos(&mut self) -> Self;
    fn ln(&mut self) -> Self;
    fn exp(&mut self) -> Self;
    fn sigmoid(&mut self) -> Self;
    fn tanh(&mut self) -> Self;
    fn square(&mut self) -> Self;
}

pub trait Tensor: RandomInit + Params + Default + ShapedArray + Activations {
    fn with_grad(data: Array<f32, Self::Dimension>, grad: Option<Grad>) -> Self;

    fn grad(&self) -> &Option<Grad>;
    fn mut_grad(&mut self) -> &mut Option<Grad>;

    fn gradient_ref(&self) -> GradientRef {
        self.grad().as_ref().unwrap().gradient_ref
    }

    fn take_tape(&mut self) -> Option<Box<GradientTape>> {
        self.mut_grad()
            .as_mut()
            .map(|grad| grad.tape.take())
            .flatten()
    }

    fn backward(&mut self) -> Option<Box<GradientTape>> {
        self.mut_grad().as_mut().map(|grad| {
            let mut tape = grad.tape.take().unwrap();
            tape.backward(grad.gradient_ref);
            tape
        })
    }

    fn keep_tape(&mut self, mut tape: Box<GradientTape>) {
        let grad = self
            .mut_grad()
            .get_or_insert_with(|| Grad::new(tape.store_gradient(Self::SHAPE)));
        grad.tape = Some(tape);
    }
}

pub trait Batch {
    type Batched<const B: usize>: Tensor;
}

pub trait Module: RandomInit + Params + Default {
    type Input: Tensor + Batch;
    type Output: Tensor + Batch;

    fn forward<const B: usize>(
        &mut self,
        input: &mut <Self::Input as Batch>::Batched<B>,
    ) -> <Self::Output as Batch>::Batched<B>;
}

pub trait Optimizer<M: Module>: DerefMut<Target = M> {
    fn step<T: Tensor>(&mut self, loss: &mut T);

    fn forward_with_derivatives<const B: usize>(
        &mut self,
        input: &mut <M::Input as Batch>::Batched<B>,
    ) -> <M::Output as Batch>::Batched<B> {
        // put tape in input
        *input.mut_grad() = None;
        input.keep_tape(Box::new(GradientTape::new()));

        // go!
        self.forward(input)
    }
}
