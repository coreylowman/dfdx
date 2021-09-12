use super::traits::Module;
use crate::gradients::{traits::Params, GradientTape};
use crate::tensor::traits::{Batch, Randomize};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Distribution;

#[derive(Default, Debug)]
pub struct ModuleChain<M1: Module, M2: Module<Input = M1::Output>> {
    first: M1,
    second: M2,
}

impl<M1: Module, M2: Module<Input = M1::Output>> Params for ModuleChain<M1, M2> {
    fn update(&mut self, tape: &GradientTape) {
        self.first.update(tape);
        self.second.update(tape);
    }
}

impl<M1: Module, M2: Module<Input = M1::Output>> Randomize for ModuleChain<M1, M2> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        self.first.randomize(rng, dist);
        self.second.randomize(rng, dist);
    }
}

impl<M1: Module, M2: Module<Input = M1::Output>> Module for ModuleChain<M1, M2> {
    type Input = M1::Input;
    type Output = M2::Output;

    fn forward<const B: usize>(
        &mut self,
        input: &mut <Self::Input as Batch>::Batched<B>,
    ) -> <Self::Output as Batch>::Batched<B> {
        self.second.forward(&mut self.first.forward(input))
    }
}
