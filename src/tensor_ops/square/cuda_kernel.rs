use crate::{shapes::Shape, tensor::Cuda, tensor_ops::ops::UnaryKernel};

impl UnaryKernel<super::SquareKernelOp, f32> for Cuda {
    fn forward<S: Shape>(
        &self,
        op: super::SquareKernelOp,
        inp: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, f32>, Self::Err> {
        todo!()
    }
    fn backward<S: Shape>(
        &self,
        op: super::SquareKernelOp,
        inp: &Self::Storage<S, f32>,
        grad_inp: &mut Self::Storage<S, f32>,
        grad_out: &Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}
