use crate::module_collection;
use crate::tensor::{Tensor1D, Tensor2D};
use crate::traits::Module;

#[derive(Default, Debug)]
pub struct Linear<const I: usize, const O: usize> {
    weight: Tensor2D<I, O>,
    bias: Tensor1D<O>,
}

module_collection!([const I: usize, const O: usize] [I, O] Linear[weight bias]);

impl<const I: usize, const O: usize> Module for Linear<I, O> {
    type Input<const B: usize> = Tensor2D<B, I>;
    type Output<const B: usize> = Tensor2D<B, O>;

    fn forward<const B: usize>(&mut self, input: &mut Self::Input<B>) -> Self::Output<B> {
        let mut ax = input * &mut self.weight;
        let ax_plus_b = &mut ax + &mut self.bias;
        ax_plus_b
    }
}
