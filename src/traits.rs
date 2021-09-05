use crate::gradients::{Grad, GradientRef, GradientTape};
use ndarray::{Array, Dimension, ShapeBuilder};
use ndarray_rand::rand::prelude::*;
use std::ops::DerefMut;

pub trait Params {
    fn randomize<R: Rng>(&mut self, rng: &mut R);
    fn register(&mut self, tape: &mut GradientTape);
    fn update(&mut self, tape: &GradientTape);
}

pub trait ShapedArray {
    type Dimension: Dimension;
    type Shape: ShapeBuilder<Dim = Self::Dimension>;
    const SHAPE: Self::Shape;

    fn data(&self) -> &Array<f32, Self::Dimension>;
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension>;
}

pub trait Tensor: Params + Default + ShapedArray {
    fn grad(&self) -> &Option<Grad>;
    fn mut_grad(&mut self) -> &mut Option<Grad>;

    fn gradient_ref(&self) -> GradientRef {
        self.grad().as_ref().unwrap().gradient_ref
    }

    fn take_tape(&mut self) -> Option<Box<GradientTape>> {
        self.mut_grad()
            .as_mut()
            .map(|grad| grad.take_tape())
            .flatten()
    }

    fn backward(&mut self) -> Option<Box<GradientTape>> {
        self.mut_grad().as_mut().map(|grad| {
            let mut tape = grad.take_tape().unwrap();
            tape.backward(grad.gradient_ref);
            tape
        })
    }

    fn keep_tape(&mut self, mut tape: Box<GradientTape>) {
        let grad = self
            .mut_grad()
            .get_or_insert_with(|| Grad::new(tape.store_gradient(Self::SHAPE)));
        grad.keep_tape(tape);
    }
}

pub trait Module: Params + Default {
    type Input<const B: usize>: Tensor;
    type Output<const B: usize>: Tensor;

    fn forward<const B: usize>(&mut self, input: &mut Self::Input<B>) -> Self::Output<B>;
}

pub trait Optimizer<M: Module>: DerefMut<Target = M> {
    fn step<T: Tensor>(&mut self, loss: &mut T);

    fn forward_with_derivatives<const B: usize>(
        &mut self,
        input: &mut M::Input<B>,
    ) -> M::Output<B> {
        let mut tape = Box::new(GradientTape::new());

        // register module's params
        self.register(&mut tape);

        // put tape in input
        *input.mut_grad() = None;
        input.keep_tape(tape);

        // go!
        self.forward(input)
    }
}
