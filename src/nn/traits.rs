use crate::prelude::{OnGradientTape, Tensor0D};
use std::ops::DerefMut;

pub trait Module<I>: Default + OnGradientTape {
    type Output;
    fn forward(&self, input: &I) -> Self::Output;
}

pub trait Optimizer<M>: DerefMut<Target = M> {
    fn step(&mut self, loss: &Tensor0D);
}
