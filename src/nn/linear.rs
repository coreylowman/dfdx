use super::module::{Init, Module};
use crate::gradients::{GradientTape, Taped};
use crate::tensor::{Batch, Randomize, Tensor1D, Tensor2D};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Uniform;

#[derive(Default, Debug)]
pub struct Linear<const I: usize, const O: usize> {
    weight: Tensor2D<I, O>,
    bias: Tensor1D<O>,
}

impl<const I: usize, const O: usize> Taped for Linear<I, O> {
    fn update(&mut self, tape: &GradientTape) {
        self.weight.update(tape);
        self.bias.update(tape);
    }
}

impl<const I: usize, const O: usize> Init for Linear<I, O> {
    fn init<R: Rng>(&mut self, rng: &mut R) {
        self.weight.randomize(rng, &Uniform::new(-1.0, 1.0));
        self.bias.randomize(rng, &Uniform::new(-1.0, 1.0));
    }
}

impl<const I: usize, const O: usize> Module for Linear<I, O> {
    type Input = Tensor1D<I>;
    type Output = Tensor1D<O>;

    fn forward<const B: usize>(
        &mut self,
        input: &mut <Self::Input as Batch>::Batched<B>,
    ) -> <Self::Output as Batch>::Batched<B> {
        &mut (input * &mut self.weight) + &mut self.bias
    }
}
