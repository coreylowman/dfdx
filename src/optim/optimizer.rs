use crate::gradients::{Gradient, GradientTape};
use crate::nn::module::Module;
use crate::tensor::*;
use std::ops::DerefMut;

pub trait Optimizer<M, I, O>: DerefMut<Target = M>
where
    M: Module<I, O>,
    I: Tensor,
    O: Tensor,
{
    fn step<T: Tensor>(&mut self, loss: &mut T);

    fn forward_with_derivatives(&mut self, input: &mut I) -> O {
        // make a new gradient tape
        let mut tape = Box::new(GradientTape::new());

        // register the input on the tape
        let gradient_ref = tape.register_gradient(<I as IsShapedArray>::SHAPE);

        // stick the tape in the input
        *input.mut_grad() = Some(Gradient::with_tape(gradient_ref, tape));

        // go!
        self.forward(input)
    }
}
