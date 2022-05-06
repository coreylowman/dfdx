use crate::prelude::{CanUpdateWithGradients, Gradients};

pub trait Module<Input>: Default + CanUpdateWithGradients {
    type Output;
    fn forward(&self, input: Input) -> Self::Output;
}

pub trait Optimizer {
    fn update<M: CanUpdateWithGradients>(&mut self, module: &mut M, gradients: Gradients);
}
