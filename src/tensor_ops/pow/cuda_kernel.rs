use crate::{shapes::Shape, tensor::Cuda, tensor_ops::ops::UnaryKernel};

impl UnaryKernel<super::PowKernelOp<i32>, f32> for Cuda {
    fn forward<S: Shape>(
        &self,
        op: super::PowKernelOp<i32>,
        inp: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, f32>, Self::Err> {
        todo!()
    }
    fn backward<S: Shape>(
        &self,
        op: super::PowKernelOp<i32>,
        inp: &Self::Storage<S, f32>,
        grad_inp: &mut Self::Storage<S, f32>,
        grad_out: &Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::device::AsKernelParam for super::PowKernelOp<f32> {}

impl UnaryOpCudaKernel for super::PowKernelOp<f32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/powf.ptx"));
    const MODULE_NAME: &'static str = "powf";
    const FWD_FN_NAME: &'static str = "powf_forward";
    const BWD_FN_NAME: &'static str = "powf_backward";
}
