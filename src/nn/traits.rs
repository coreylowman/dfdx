use crate::{
    gradients::GradientTape,
    prelude::{HasGradients, Tensor0D},
};

pub trait Module<I>: Default + HasGradients {
    type Output;
    fn forward(&self, input: &I) -> Self::Output;
}

pub trait Optimizer {
    fn compute_gradients(&mut self, loss: &Tensor0D) -> GradientTape;
}
