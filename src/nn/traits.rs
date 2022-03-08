use crate::gradients::GradientTape;
use crate::prelude::OnGradientTape;
use crate::tensor::*;
use std::ops::DerefMut;

pub trait Module<I>: Default + OnGradientTape {
    type Output;

    fn forward(&mut self, input: &mut I) -> Self::Output;

    fn forward_with_derivatives(&mut self, input: &mut I) -> Self::Output
    where
        I: Tensor,
    {
        // make a new gradient tape
        let mut tape = GradientTape::new();

        // register gradient on tape, and put tape into resulting gradient
        *input.mut_grad_ref() = Some(tape.allocate_gradient(I::SHAPE));
        *input.mut_tape() = Some(Box::new(tape));

        // go!
        self.forward(input)
    }
}

pub trait Optimizer<M>: DerefMut<Target = M> {
    fn step(&mut self, tape: GradientTape);
}
