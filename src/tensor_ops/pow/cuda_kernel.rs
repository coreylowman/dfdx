use super::PowKernelOp;
use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::PowKernelOp<f32> {}

impl UnaryOpCudaKernel for super::PowKernelOp<f32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/pow.ptx"));
    const MODULE_NAME: &'static str = "pow";
    const FWD_FN_NAME: &'static str = "pow_forward";
    const BWD_FN_NAME: &'static str = "pow_backward";
}

use crate::{shapes::Shape, tensor::cuda::Cuda, tensor_ops::ops::UnaryKernel};

impl UnaryKernel<PowKernelOp<i32>, f32> for Cuda {
    fn forward<S: Shape>(
        &self,
        op: PowKernelOp<i32>,
        inp: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, f32>, Self::Err> {
        self.forward(PowKernelOp(op.0 as f32), inp)
    }

    fn backward<S: Shape>(
        &self,
        op: PowKernelOp<i32>,
        inp: &Self::Storage<S, f32>,
        grad_inp: &mut Self::Storage<S, f32>,
        grad_out: &Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        self.backward(PowKernelOp(op.0 as f32), inp, grad_inp, grad_out)
    }
}
