use crate::module_collection;
use crate::tensor::{Tensor1D, Tensor2D};
use crate::traits::{Batch, Module};

#[derive(Default, Debug)]
pub struct Linear<const I: usize, const O: usize> {
    weight: Tensor2D<I, O>,
    bias: Tensor1D<O>,
}

module_collection!([const I: usize, const O: usize], [I, O], Linear, [weight, bias, ]);

impl<const I: usize, const O: usize> Module for Linear<I, O> {
    type Input = Tensor1D<I>;
    type Output = Tensor1D<O>;

    fn forward<const B: usize>(
        &mut self,
        input: &mut <Self::Input as Batch>::Batched<B>,
    ) -> <Self::Output as Batch>::Batched<B> {
        let mut ax = input * &mut self.weight;
        let ax_plus_b = &mut ax + &mut self.bias;
        ax_plus_b
    }
}
