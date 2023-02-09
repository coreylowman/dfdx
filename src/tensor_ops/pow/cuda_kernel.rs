use crate::{
    shapes::Shape,
    tensor::cuda::Cuda,
    tensor_ops::{cuda_kernels::UnaryOpCudaKernel, ops::UnaryKernel},
};

unsafe impl cudarc::driver::AsKernelParam for super::PowfKernelOp<f32> {}
unsafe impl cudarc::driver::AsKernelParam for super::PowfKernelOp<f64> {}

impl UnaryOpCudaKernel<f32> for super::PowfKernelOp<f32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/pow.ptx"));
    const MODULE_NAME: &'static str = "pow";
    const FWD_FN_NAME: &'static str = "pow_forward_f32";
    const BWD_FN_NAME: &'static str = "pow_backward_f32";
}

impl UnaryOpCudaKernel<f64> for super::PowfKernelOp<f64> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/pow.ptx"));
    const MODULE_NAME: &'static str = "pow";
    const FWD_FN_NAME: &'static str = "pow_forward_f64";
    const BWD_FN_NAME: &'static str = "pow_backward_f64";
}

impl UnaryKernel<super::PowiKernelOp, f32> for Cuda {
    fn forward<S: Shape>(
        &self,
        op: super::PowiKernelOp,
        inp: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, f32>, Self::Err> {
        self.forward(super::PowfKernelOp(op.0 as f32), inp)
    }

    fn backward<S: Shape>(
        &self,
        op: super::PowiKernelOp,
        inp: &Self::Storage<S, f32>,
        grad_inp: &mut Self::Storage<S, f32>,
        grad_out: &Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        self.backward(super::PowfKernelOp(op.0 as f32), inp, grad_inp, grad_out)
    }
}

impl UnaryKernel<super::PowiKernelOp, f64> for Cuda {
    fn forward<S: Shape>(
        &self,
        op: super::PowiKernelOp,
        inp: &Self::Storage<S, f64>,
    ) -> Result<Self::Storage<S, f64>, Self::Err> {
        self.forward(super::PowfKernelOp(op.0 as f64), inp)
    }

    fn backward<S: Shape>(
        &self,
        op: super::PowiKernelOp,
        inp: &Self::Storage<S, f64>,
        grad_inp: &mut Self::Storage<S, f64>,
        grad_out: &Self::Storage<S, f64>,
    ) -> Result<(), Self::Err> {
        self.backward(super::PowfKernelOp(op.0 as f64), inp, grad_inp, grad_out)
    }
}
