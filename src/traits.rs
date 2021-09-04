use crate::gradients::{Grad, GradientRef, GradientTape};
use ndarray::{Array, Dimension};
use ndarray_rand::rand::prelude::*;
use std::ops::DerefMut;

pub trait Params {
    fn randomize<R: Rng>(&mut self, rng: &mut R);
    fn register(&mut self, tape: &mut GradientTape);
    fn update(&mut self, tape: &GradientTape);
}

pub trait Tensor: Params + Default {
    type Dimension: Dimension;
    const SHAPE: &'static [usize];

    fn grad(&self) -> &Option<Grad>;
    fn mut_grad(&mut self) -> &mut Option<Grad>;
    fn data(&self) -> &Array<f32, Self::Dimension>;
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension>;

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

impl<T> Params for T
where
    T: Tensor,
{
    fn randomize<R: Rng>(&mut self, rng: &mut R) {
        self.mut_data().map_inplace(|f| *f = rng.gen())
    }

    fn register(&mut self, tape: &mut GradientTape) {
        if self.grad().is_none() {
            let gradient_ref = tape.store_gradient(Self::SHAPE);
            *self.mut_grad() = Some(Grad::new(gradient_ref));
        }
    }

    fn update(&mut self, tape: &GradientTape) {
        assert!(self.grad().is_some());
        let grad = self.mut_grad().as_mut().unwrap();
        let gradient = &tape[grad.gradient_ref];
        *self.mut_data() -= gradient;
        *self.mut_grad() = None;
    }
}

pub trait Module: Params + Default {
    type Input: Tensor;
    type Output: Tensor;

    fn forward(&mut self, input: &mut Self::Input) -> Self::Output;
}

pub trait Optimizer<M: Module>: DerefMut<Target = M> {
    fn step<T: Tensor>(&mut self, loss: &mut T);

    fn forward_with_derivatives(&mut self, input: &mut M::Input) -> M::Output {
        let mut tape = GradientTape::new();

        // register module's params
        self.register(&mut tape);

        // put tape in input
        *input.mut_grad() = None;
        input.register(&mut tape);
        input.keep_tape(Box::new(tape));

        // go!
        self.forward(input)
    }
}
