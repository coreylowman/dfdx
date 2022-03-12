use crate::gradients::GradientTape;
use crate::prelude::OnGradientTape;
use std::ops::DerefMut;

pub trait Module<I>: Default + OnGradientTape {
    type Output;

    fn forward(&self, input: &I) -> Self::Output;
}

pub trait Optimizer<M>: DerefMut<Target = M> {
    fn step(&mut self, tape: GradientTape);
}
