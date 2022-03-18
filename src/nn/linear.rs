use super::traits::Module;
use crate::gradients::GradientTape;
use crate::prelude::{
    add_no_tape, add_with_tape, broadcast_add_no_tape, broadcast_add_with_tape, matmat_mul_no_tape,
    matmat_mul_with_tape, vecmat_mul_no_tape, vecmat_mul_with_tape, NoTape, WithTape,
};
use crate::tensor::{CanUpdateWithTape, Randomize, Tensor1D, Tensor2D};
use rand::{distributions::Distribution, Rng};

#[derive(Default, Debug)]
pub struct Linear<const I: usize, const O: usize> {
    weight: Tensor2D<I, O, NoTape>,
    bias: Tensor1D<O, NoTape>,
}

impl<const I: usize, const O: usize> CanUpdateWithTape for Linear<I, O> {
    fn update_with_tape(&mut self, tape: &GradientTape) {
        self.weight.update_with_tape(tape);
        self.bias.update_with_tape(tape);
    }
}

impl<const I: usize, const O: usize> Randomize for Linear<I, O> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        self.weight.randomize(rng, dist);
        self.bias.randomize(rng, dist);
    }
}

// 1d forward
impl<const I: usize, const O: usize> Module<Tensor1D<I, NoTape>> for Linear<I, O> {
    type Output = Tensor1D<O, NoTape>;

    fn forward(&self, x: Tensor1D<I, NoTape>) -> Self::Output {
        add_no_tape(&self.bias, vecmat_mul_no_tape(x, &self.weight))
    }
}

impl<const I: usize, const O: usize> Module<Tensor1D<I, WithTape>> for Linear<I, O> {
    type Output = Tensor1D<O, WithTape>;

    fn forward(&self, x: Tensor1D<I, WithTape>) -> Self::Output {
        add_with_tape(&self.bias, vecmat_mul_with_tape(x, &self.weight))
    }
}

// Batched 2d forward
impl<const B: usize, const I: usize, const O: usize> Module<Tensor2D<B, I, NoTape>>
    for Linear<I, O>
{
    type Output = Tensor2D<B, O, NoTape>;
    fn forward(&self, x: Tensor2D<B, I, NoTape>) -> Self::Output {
        broadcast_add_no_tape(matmat_mul_no_tape(x, &self.weight), &self.bias)
    }
}

impl<const B: usize, const I: usize, const O: usize> Module<Tensor2D<B, I, WithTape>>
    for Linear<I, O>
{
    type Output = Tensor2D<B, O, WithTape>;
    fn forward(&self, x: Tensor2D<B, I, WithTape>) -> Self::Output {
        broadcast_add_with_tape(matmat_mul_with_tape(x, &self.weight), &self.bias)
    }
}
