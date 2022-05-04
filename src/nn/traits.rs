use crate::prelude::{CanUpdateWithGradients, Gradients, Tensor0D, WithTape};

pub trait Module<Input>: Default + CanUpdateWithGradients {
    type Output;
    fn forward(&self, input: Input) -> Self::Output;
}

pub trait Optimizer {
    fn compute_gradients(&mut self, loss: Tensor0D<WithTape>) -> (f32, Gradients);
}
