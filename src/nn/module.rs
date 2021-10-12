use crate::gradients::traits::Taped;
use crate::tensor::{Batch, Tensor};
use ndarray_rand::rand::Rng;

pub trait Init {
    fn init<R: Rng>(&mut self, rng: &mut R);
}

pub trait Module: Init + Taped + Default {
    type Input: Tensor + Batch;
    type Output: Tensor + Batch;

    fn forward<const B: usize>(
        &mut self,
        input: &mut <Self::Input as Batch>::Batched<B>,
    ) -> <Self::Output as Batch>::Batched<B>;
}
