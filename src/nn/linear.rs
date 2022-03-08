use super::traits::Module;
use crate::gradients::GradientTape;
use crate::tensor::{
    add, broadcast_add, matmat_mul, vecmat_mul, OnGradientTape, Randomize, Tensor1D, Tensor2D,
};
use rand::{distributions::Distribution, Rng};

#[derive(Default, Debug)]
pub struct Linear<const I: usize, const O: usize> {
    weight: Tensor2D<I, O>,
    bias: Tensor1D<O>,
}

impl<const I: usize, const O: usize> OnGradientTape for Linear<I, O> {
    fn put_on(&mut self, tape: &mut GradientTape) {
        self.weight.put_on(tape);
        self.bias.put_on(tape);
    }

    fn update_with(&mut self, tape: &GradientTape) {
        self.weight.update_with(tape);
        self.bias.update_with(tape);
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

    fn forward(&mut self, x: &mut Tensor1D<I>) -> Self::Output {
        add(&mut vecmat_mul(x, &mut self.weight), &mut self.bias)
    }
}

// Batched 2d forward
impl<const B: usize, const I: usize, const O: usize> Module<Tensor2D<B, I>> for Linear<I, O> {
    type Output = Tensor2D<B, O>;
    fn forward(&mut self, x: &mut Tensor2D<B, I>) -> Self::Output {
        broadcast_add(&mut matmat_mul(x, &mut self.weight), &mut self.bias)
    }
}
