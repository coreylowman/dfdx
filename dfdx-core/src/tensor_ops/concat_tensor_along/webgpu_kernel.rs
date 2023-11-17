use crate::{shapes::*, tensor::*};

impl<E: Dtype> super::ConcatAlongKernel<E> for Webgpu {
    fn forward<A: Shape, B: Shape, C: Shape>(
        &self,
        ax: usize,
        a: &Tensor<A, E, Self>,
        b: &Tensor<B, E, Self>,
        c: &mut Tensor<C, E, Self>,
    ) -> Result<(), Error> {
        todo!()
    }

    fn backward<A: Shape, B: Shape>(
        &self,
        ax: usize,
        a: &GhostTensor<A, E, Self>,
        grad_a: &mut Self::Vec,
        b: &GhostTensor<B, E, Self>,
        grad_b: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        todo!()
    }
}
