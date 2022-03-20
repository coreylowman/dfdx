use super::traits::Module;
use crate::gradients::GradientTape;
use crate::prelude::{add, broadcast_add, matmat_mul, vecmat_mul, NoTape, TapeManager};
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
impl<const I: usize, const O: usize, Mgr: TapeManager> Module<Tensor1D<I, Mgr>> for Linear<I, O> {
    type Output = Tensor1D<O, Mgr>;

    fn forward(&self, x: Tensor1D<I, Mgr>) -> Self::Output {
        add(&self.bias, vecmat_mul(x, &self.weight))
    }
}

// Batched 2d forward
impl<const B: usize, const I: usize, const O: usize, Mgr: TapeManager> Module<Tensor2D<B, I, Mgr>>
    for Linear<I, O>
{
    type Output = Tensor2D<B, O, Mgr>;
    fn forward(&self, x: Tensor2D<B, I, Mgr>) -> Self::Output {
        broadcast_add(matmat_mul(x, &self.weight), &self.bias)
    }
}
