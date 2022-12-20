use crate::{shapes::Shape, tensor::Cuda, tensor_ops::ops::BinaryKernel};

impl BinaryKernel<super::MinimumKernelOp, f32> for Cuda {
    fn forward<S: Shape>(
        &self,
        op: super::MinimumKernelOp,
        lhs: &Self::Storage<S, f32>,
        rhs: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, f32>, Self::Err> {
        todo!()
    }
    fn backward<S: Shape>(
        &self,
        op: super::MinimumKernelOp,
        lhs: &Self::Storage<S, f32>,
        grad_lhs: &mut Self::Storage<S, f32>,
        rhs: &Self::Storage<S, f32>,
        grad_rhs: &mut Self::Storage<S, f32>,
        grad_out: &Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}
