use crate::gradients::Taped;
use crate::tensor::Tensor;
use ndarray_rand::rand::Rng;

pub trait Init {
    fn init<R: Rng>(&mut self, rng: &mut R);
}

pub trait Module<I, O>: Init + Taped + Default
where
    I: Tensor,
    O: Tensor,
{
    fn forward(&mut self, input: &mut I) -> O;
}
