use ndarray_rand::{rand::Rng, rand_distr::Distribution};

use crate::{gradients::GradientTape, traits::*};
use std::marker::PhantomData;

#[derive(Debug, Default)]
pub struct ReLU<T: Tensor + Batch> {
    marker: PhantomData<T>,
}

impl<T: Tensor + Batch> RandomInit for ReLU<T> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, _rng: &mut R, _dist: &D) {}
}

impl<T: Tensor + Batch> Params for ReLU<T> {
    fn register(&mut self, _tape: &mut GradientTape) {}
    fn update(&mut self, _tape: &GradientTape) {}
}

impl<T: Tensor + Batch> Module for ReLU<T> {
    type Input = T;
    type Output = T;

    fn forward<const B: usize>(&mut self, input: &mut T::Batched<B>) -> T::Batched<B> {
        input.relu()
    }
}

#[derive(Debug, Default)]
pub struct Sin<T: Tensor + Batch> {
    marker: PhantomData<T>,
}

impl<T: Tensor + Batch> RandomInit for Sin<T> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, _rng: &mut R, _dist: &D) {}
}

impl<T: Tensor + Batch> Params for Sin<T> {
    fn register(&mut self, _tape: &mut GradientTape) {}
    fn update(&mut self, _tape: &GradientTape) {}
}

impl<T: Tensor + Batch> Module for Sin<T> {
    type Input = T;
    type Output = T;

    fn forward<const B: usize>(&mut self, input: &mut T::Batched<B>) -> T::Batched<B> {
        input.sin()
    }
}
