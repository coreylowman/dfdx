use super::traits::Module;
use crate::gradients::{traits::Params, GradientTape};
use crate::tensor::traits::{Batch, Randomize};
use crate::tensor::{Tensor1D, Tensor2D};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Distribution;

#[derive(Default, Debug)]
pub struct Linear<const I: usize, const O: usize> {
    weight: Tensor2D<I, O>,
    bias: Tensor1D<O>,
}

impl<const I: usize, const O: usize> Params for Linear<I, O> {
    fn update(&mut self, tape: &GradientTape) {
        self.weight.update(tape);
        self.bias.update(tape);
    }
}

impl<const I: usize, const O: usize> Randomize for Linear<I, O> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        self.weight.randomize(rng, dist);
        self.bias.randomize(rng, dist);
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
