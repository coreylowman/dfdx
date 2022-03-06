use super::traits::Module;
use crate::gradients::{GradientTape, OnGradientTape};
use crate::tensor::{Randomize, Tensor1D, Tensor2D};
use rand::{distributions::Distribution, Rng};

#[derive(Default, Debug)]
pub struct Linear<const I: usize, const O: usize> {
    weight: Tensor2D<I, O>,
    bias: Tensor1D<O>,
}

impl<const I: usize, const O: usize> OnGradientTape for Linear<I, O> {
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

// 1d forward
impl<const I: usize, const O: usize> Module<Tensor1D<I>> for Linear<I, O> {
    type Output = Tensor1D<O>;

    fn forward(&mut self, input: &mut Tensor1D<I>) -> Self::Output {
        &mut (input * &mut self.weight) + &mut self.bias
    }
}

// Batched 2d forward
impl<const B: usize, const I: usize, const O: usize> Module<Tensor2D<B, I>> for Linear<I, O> {
    type Output = Tensor2D<B, O>;
    fn forward(&mut self, input: &mut Tensor2D<B, I>) -> Self::Output {
        &mut (input * &mut self.weight) + &mut self.bias
    }
}
