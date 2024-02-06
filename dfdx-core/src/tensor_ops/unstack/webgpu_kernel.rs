use crate::{prelude::NoneTape, shapes::*, tensor::Webgpu};
use std::vec::Vec;

impl<E: Dtype> super::UnstackKernel<E> for Webgpu {
    fn forward<Items, S: Shape, T: Tape<E, Self>>(
        &self,
        stack: Tensor<S, E, Self, NoneTape>,
    ) -> Result<Items, Error>
    where
        S: super::SubDim,
        Items: Array<Tensor<S::Tail, E, Self, T>, Dim = S::Head>,
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
