use super::traits::Module;
use crate::prelude::*;
use rand::{distributions::Distribution, Rng};

impl<F: DifferentiableFunction> HasGradients for F {
    fn update_with_gradients(&mut self, _: &GradientTape) {}
}

impl<F: DifferentiableFunction> Randomize for F {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, _: &mut R, _: &D) {}
}

impl<F: DifferentiableFunction + Default, I: Tensor> Module<I> for F {
    type Output = I;
    fn forward(&self, input: &I) -> Self::Output {
        input.apply::<F>()
    }
}
