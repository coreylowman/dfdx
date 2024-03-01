use crate::{shapes::*, tensor::*};

impl<E: Dtype> super::ConcatAlongKernel<E> for Webgpu {
    fn forward<AB: Shape, A: Shape, B: Shape>(
        &self,
        _ax: usize,
        _ab: &Tensor<AB, E, Self>,
        _a: &mut Tensor<A, E, Self>,
        _b: &mut Tensor<B, E, Self>,
    ) -> Result<(), Error> {
        todo!()
    }

    fn backward<AB: Shape, A: Shape, B: Shape>(
        &self,
        _ax: usize,
        _ab: &GhostTensor<AB, E, Self>,
        _grad_ab: &mut Self::Vec,
        _a: &GhostTensor<A, E, Self>,
        _b: &GhostTensor<B, E, Self>,
        _a_or_b: AorB,
        _grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        todo!()
    }
}
