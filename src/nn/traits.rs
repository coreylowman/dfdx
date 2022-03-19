use crate::{
    gradients::GradientTape,
    prelude::{CanUpdateWithTape, Tensor0D, WithTape},
};

pub trait Module<Input>: Default + CanUpdateWithTape {
    type Output;
    fn forward(&self, input: Input) -> Self::Output;
}

pub trait Optimizer {
    fn compute_gradients(&mut self, loss: Tensor0D<WithTape>) -> (f32, Box<GradientTape>);
}
