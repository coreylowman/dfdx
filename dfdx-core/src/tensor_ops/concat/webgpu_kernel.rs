use crate::{shapes::*, tensor::*};

use super::ConcatShape;

impl<E: Dtype> super::ConcatKernel<E> for Webgpu {
    fn forward<A: Shape, B: Shape>(
        &self,
        a: &Tensor<A, E, Self>,
        b: &Tensor<B, E, Self>,
    ) -> Result<Tensor<A::Catted, E, Self>, Error>
    where
        A: ConcatShape<B>,
    {
        todo!()
    }

    fn backward(
        &self,
        grad_a: &mut Self::Vec,
        grad_b: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        todo!()
    }
}
