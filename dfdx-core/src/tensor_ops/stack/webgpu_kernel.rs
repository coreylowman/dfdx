use crate::{shapes::*, tensor::Webgpu};
use std::vec::Vec;

impl<E: Dtype> super::StackKernel<E> for Webgpu {
    fn forward<S: Shape, Num: Dim>(
        &self,
        num: Num,
        inp: &[crate::prelude::Tensor<S, E, Self>],
    ) -> Result<crate::prelude::Tensor<S::Larger, E, Self>, crate::prelude::Error>
    where
        S: crate::prelude::AddDim<Num>,
    {
        todo!()
    }

    fn backward(
        &self,
        grad_inp: Vec<&mut Self::Vec>,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
