use crate::prelude::CanUpdateWithGradients;

/// A unit of a neural network. Acts on the generic `Input`
/// and produces `Module::Output`.
///
/// Generic `Input` means you can implement module for multiple
/// input types on the same struct. For example [Linear] implements
/// [Module] for 1d inputs and 2d inputs.
pub trait Module<Input>: ResetParams + CanUpdateWithGradients {
    type Output;
    fn forward(&self, input: Input) -> Self::Output;
}

pub trait ResetParams {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R);
}
