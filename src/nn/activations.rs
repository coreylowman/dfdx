use super::traits::Module;
use crate::{
    diff_fns::{ApplyDifferentiableFunction, DifferentiableFunction},
    gradients::GradientTape,
    tensor::{HasGradients, Randomize, Tensor},
};
use rand::{distributions::Distribution, Rng};

impl<F> HasGradients for F
where
    F: DifferentiableFunction,
{
    fn update_with_gradients(&mut self, _: &GradientTape) {}
}

impl<F> Randomize for F
where
    F: DifferentiableFunction,
{
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, _: &mut R, _: &D) {}
}

impl<F, I> Module<I> for F
where
    I: Tensor,
    F: DifferentiableFunction + Default,
{
    type Output = I;

    fn forward(&self, input: &I) -> Self::Output {
        input.apply::<F>()
    }
}
