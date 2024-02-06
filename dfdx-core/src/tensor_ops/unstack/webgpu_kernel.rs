use crate::{prelude::NoneTape, shapes::*, tensor::Webgpu};
use std::vec::Vec;

impl<E: Dtype, Items> super::UnstackKernel<E, Items> for Webgpu {
    fn forward<S: Shape>(&self, stack: Tensor<S, E, Self, NoneTape>) -> Result<Items, Error>
    where
        S: super::SubDim,
        Items: Array<Option<Tensor<S::Tail, E, Self, NoneTape>>, Dim = S::Head>,
    {
        todo!()
    }
    fn backward(
        &self,
        grad_stack: &mut Self::Vec,
        grad_unstack: &Self::Vec,
        unstack_idx: usize,
    ) -> Result<(), Error> {
        todo!()
    }
}
