use crate::gradients::GradientTape;
use crate::tensor::*;
use std::ops::DerefMut;

pub trait Module<I>: Default {
    type Output;

    fn forward(&mut self, input: &mut I) -> Self::Output;

    fn forward_with_derivatives(&mut self, input: &mut I) -> Self::Output
    where
        I: Tensor,
    {
        // make a new gradient tape
        let mut tape = GradientTape::new();

        // register the input on the tape
        let mut gradient = tape.allocate_gradient(I::SHAPE);

        // move ownership of tape into this gradient
        gradient.tape = Some(Box::new(tape));

        // register gradient on tape, and put tape into resulting gradient
        *input.mut_grad() = Some(gradient);

        // go!
        self.forward(input)
    }
}

pub trait Optimizer<M>: DerefMut<Target = M> {
    fn step(&mut self, tape: GradientTape);
}
