use super::traits::{Init, Module};
use crate::{
    gradients::{traits::Params, GradientTape},
    tensor::traits::{Activations, Batch, Tensor},
};
use ndarray_rand::rand::Rng;
use std::marker::PhantomData;

#[derive(Debug, Default)]
pub struct ReLU<T: Tensor + Batch> {
    marker: PhantomData<T>,
}

impl<T: Tensor + Batch> Init for ReLU<T> {
    fn init<R: Rng>(&mut self, _rng: &mut R) {}
}

impl<T: Tensor + Batch> Params for ReLU<T> {
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

impl<T: Tensor + Batch> Init for Sin<T> {
    fn init<R: Rng>(&mut self, _rng: &mut R) {}
}

impl<T: Tensor + Batch> Params for Sin<T> {
    fn update(&mut self, _tape: &GradientTape) {}
}

impl<T: Tensor + Batch> Module for Sin<T> {
    type Input = T;
    type Output = T;

    fn forward<const B: usize>(&mut self, input: &mut T::Batched<B>) -> T::Batched<B> {
        input.sin()
    }
}
