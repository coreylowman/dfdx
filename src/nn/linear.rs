use crate::gradients::GradientTape;
use crate::tensor::{Tensor1D, Tensor2D};
use crate::traits::{Module, Params, RandomInit};
use ndarray_rand::rand::Rng;

#[derive(Default, Debug)]
pub struct Linear<const I: usize, const O: usize> {
    weight: Tensor2D<I, O>,
    bias: Tensor1D<O>,
}

impl<const I: usize, const O: usize> RandomInit for Linear<I, O> {
    fn randomize<R: Rng>(&mut self, rng: &mut R) {
        self.weight.randomize(rng);
        self.bias.randomize(rng);
    }
}

impl<const I: usize, const O: usize> Params for Linear<I, O> {
    fn register(&mut self, tape: &mut GradientTape) {
        self.weight.register(tape);
        self.bias.register(tape);
    }

    fn update(&mut self, tape: &GradientTape) {
        self.weight.update(tape);
        self.bias.update(tape);
    }
}

impl<const I: usize, const O: usize> Module for Linear<I, O> {
    type Input<const B: usize> = Tensor2D<B, I>;
    type Output<const B: usize> = Tensor2D<B, O>;

    fn forward<const B: usize>(&mut self, input: &mut Self::Input<B>) -> Self::Output<B> {
        let mut ax = input * &mut self.weight;
        let ax_plus_b = &mut ax + &mut self.bias;
        ax_plus_b
    }
}
