use crate::gradients::{Gradient, GradientTape, HasGradient};
use crate::tensor::*;
use std::ops::DerefMut;

pub trait Module<I>: Default {
    type Output;

    fn forward(&mut self, input: &mut I) -> Self::Output;

    fn forward_with_derivatives(&mut self, input: &mut I) -> Self::Output
    where
        I: IsShapedArray + HasGradient,
    {
        // make a new gradient tape
        let mut tape = Box::new(GradientTape::new());

        // register the input on the tape
        let gradient_ref = tape.register_gradient(I::SHAPE);

        // stick the tape in the input
        *input.mut_grad() = Some(Gradient::on_tape(gradient_ref, tape));

        // go!
        self.forward(input)
    }
}

pub trait Optimizer<M>: DerefMut<Target = M> {
    fn step<T: Tensor>(&mut self, loss: &mut T);
}
