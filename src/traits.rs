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

    fn grad(&self) -> &Grad;
    fn mut_grad(&mut self) -> &mut Grad;
    fn data(&self) -> &Array<f32, Self::Dimension>;
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension>;

    fn gradient_ref(&self) -> GradientRef {
        self.grad().gradient_ref()
    }

    fn backward(&mut self) -> Box<GradientTape> {
        self.mut_grad().backward()
    }

    fn take_tape(&mut self) -> Option<Box<GradientTape>> {
        self.mut_grad().take_tape()
    }

    fn keep_tape(&mut self, tape: Option<Box<GradientTape>>) {
        self.mut_grad().keep_tape(tape);
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
        if !self.grad().is_registered() {
            self.mut_grad()
                .set_gradient_ref(tape.store_gradient(Self::SHAPE));
        }
    }

    fn update(&mut self, tape: &GradientTape) {
        let gradient = &tape[self.mut_grad().take_gradient_ref()];
        *self.mut_data() -= gradient;
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

        // register input params
        input.mut_grad().clear_gradient_ref();
        input.register(&mut tape);

        // put tape in input
        input.keep_tape(Some(Box::new(tape)));

        // go!
        self.forward(input)
    }
}
