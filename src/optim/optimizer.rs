use crate::gradients::GradientTape;
use crate::nn::module::Module;
use crate::tensor::*;
use std::ops::DerefMut;

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
