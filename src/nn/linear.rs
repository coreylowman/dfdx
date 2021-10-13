use super::module::{Init, Module};
use crate::gradients::{GradientTape, Taped};
use crate::tensor::{Randomize, Tensor1D, Tensor2D};
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

impl<const I: usize, const O: usize> Module<Tensor1D<I>, Tensor1D<O>> for Linear<I, O> {
    fn forward(&mut self, input: &mut Tensor1D<I>) -> Tensor1D<O> {
        &mut (input * &mut self.weight) + &mut self.bias
    }
}

impl<const B: usize, const I: usize, const O: usize> Module<Tensor2D<B, I>, Tensor2D<B, O>>
    for Linear<I, O>
{
    fn forward(&mut self, input: &mut Tensor2D<B, I>) -> Tensor2D<B, O> {
        &mut (input * &mut self.weight) + &mut self.bias
    }
}
