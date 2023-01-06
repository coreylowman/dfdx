use crate::{
    shapes::Shape,
    tensor::Cuda,
    tensor_ops::ops::{BinaryKernel, UnaryKernel},
};

impl UnaryKernel<super::ScalarAddKernelOp<f32>, f32> for Cuda {
    fn forward<S: Shape>(
        &self,
        op: super::ScalarAddKernelOp<f32>,
        inp: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, f32>, Self::Err> {
        todo!()
    }
    fn backward<S: Shape>(
        &self,
        op: super::ScalarAddKernelOp<f32>,
        inp: &Self::Storage<S, f32>,
        grad_inp: &mut Self::Storage<S, f32>,
        grad_out: &Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

impl BinaryKernel<super::BinaryAddKernelOp, f32> for Cuda {
    fn forward<S: Shape>(
        &self,
        op: super::BinaryAddKernelOp,
        lhs: &Self::Storage<S, f32>,
        rhs: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, f32>, Self::Err> {
        todo!()
    }
    fn backward<S: Shape>(
        &self,
        op: super::BinaryAddKernelOp,
        lhs: &Self::Storage<S, f32>,
        grad_lhs: &mut Self::Storage<S, f32>,
        rhs: &Self::Storage<S, f32>,
        grad_rhs: &mut Self::Storage<S, f32>,
        grad_out: &Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}
