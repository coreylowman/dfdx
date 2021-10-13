use crate::gradients::Taped;
use crate::tensor::Tensor;
use ndarray_rand::rand::Rng;

pub trait Init {
    fn init<R: Rng>(&mut self, rng: &mut R);
}

/*
TODO add variant of Module that accepts generic input/output parameter

this can be used for activation functions that don't care about the size of the data
they act on
*/
pub trait Module<I, O>: Init + Taped + Default
where
    I: Tensor,
    O: Tensor,
{
    fn forward(&mut self, input: &mut I) -> O;
}
