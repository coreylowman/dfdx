use super::traits::Module;
use crate::{
    diff_fns::{ApplyDifferentiableFunction, DifferentiableFunction},
    gradients::GradientTape,
    tensor::{OnGradientTape, Randomize, Tensor},
};
use rand::{distributions::Distribution, Rng};

impl<F> OnGradientTape for F
where
    F: DifferentiableFunction,
{
    fn put_on(&mut self, _: &mut GradientTape) {}
    fn update_with(&mut self, _: &GradientTape) {}
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
