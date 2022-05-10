use crate::prelude::*;
use rand::{distributions::Distribution, Rng};

#[derive(Default, Debug, Clone)]
pub struct Linear<const I: usize, const O: usize> {
    pub weight: Tensor2D<I, O, NoTape>,
    pub bias: Tensor1D<O, NoTape>,
}

impl<const I: usize, const O: usize> CanUpdateWithGradients for Linear<I, O> {
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.weight.update(grads);
        self.bias.update(grads);
    }
}

impl<const I: usize, const O: usize> Randomize for Linear<I, O> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        self.weight.randomize(rng, dist);
        self.bias.randomize(rng, dist);
    }
}

// 1d forward
impl<const I: usize, const O: usize, H: TapeHolder> Module<Tensor1D<I, H>> for Linear<I, O> {
    type Output = Tensor1D<O, H>;
    fn forward(&self, x: Tensor1D<I, H>) -> Self::Output {
        add(&self.bias, vecmat_mul(x, &self.weight))
    }
}

// Batched 2d forward
impl<const B: usize, const I: usize, const O: usize, H: TapeHolder> Module<Tensor2D<B, I, H>>
    for Linear<I, O>
{
    type Output = Tensor2D<B, O, H>;
    fn forward(&self, x: Tensor2D<B, I, H>) -> Self::Output {
        broadcast_outer_add(matmat_mul(x, &self.weight), &self.bias)
    }
}
