use crate::gradients::GradientTape;
use crate::tensor::{Tensor1D, Tensor2D};
use crate::traits::{Module, Params};
use ndarray_rand::rand::Rng;

#[derive(Default, Debug)]
pub struct Linear<const I: usize, const O: usize> {
    weight: Tensor2D<O, I>,
    bias: Tensor1D<O>,
}

impl<const I: usize, const O: usize> Params for Linear<I, O> {
    fn randomize<R: Rng>(&mut self, rng: &mut R) {
        self.weight.randomize(rng);
        self.bias.randomize(rng);
    }

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
    type Input = Tensor1D<I>;
    type Output = Tensor1D<O>;

    fn forward(&mut self, input: &mut Self::Input) -> Self::Output {
        let mut ax = &mut self.weight * input;
        let ax_plus_b = &mut ax + &mut self.bias;
        ax_plus_b
    }
}
