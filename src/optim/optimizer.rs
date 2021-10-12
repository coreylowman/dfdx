use crate::gradients::{Gradient, GradientTape, HasGradient};
use crate::nn::module::Module;
use crate::tensor::*;
use std::ops::DerefMut;

pub trait Optimizer<M: Module>: DerefMut<Target = M> {
    fn step<T: Tensor>(&mut self, loss: &mut T);

    fn forward_with_derivatives<const B: usize>(
        &mut self,
        input: &mut <M::Input as Batch>::Batched<B>,
    ) -> <M::Output as Batch>::Batched<B> {
        // make a new gradient tape
        let mut tape = Box::new(GradientTape::new());

        // register the input on the tape
        let gradient_ref =
            tape.register_gradient(<<M::Input as Batch>::Batched<B> as ShapedArray>::SHAPE);

        // stick the tape in the input
        *input.mut_grad() = Some(Gradient::with_tape(gradient_ref, tape));

        // go!
        self.forward(input)
    }
}
