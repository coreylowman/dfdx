use super::PowfKernelOp;
use crate::{
    shapes::*,
    tensor::cuda::Cuda,
    tensor_ops::{cuda_kernels::cuda_unary, ops::UnaryKernel},
};

unsafe impl cudarc::driver::AsKernelParam for super::PowfKernelOp<f32> {}
unsafe impl cudarc::driver::AsKernelParam for super::PowfKernelOp<f64> {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/pow.ptx"));

cuda_unary!(PowfKernelOp<f32>, f32, PTX, "pow_fwd_f32", "pow_bwd_f32");
cuda_unary!(PowfKernelOp<f64>, f64, PTX, "pow_fwd_f64", "pow_bwd_f64");

impl<E: Dtype> UnaryKernel<super::PowiKernelOp, E> for Cuda
where
    Self: UnaryKernel<super::PowfKernelOp<E>, E>,
{
    fn forward<S: Shape>(
        &self,
        op: super::PowiKernelOp,
        inp: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        self.forward(super::PowfKernelOp(E::from_i32(op.0).unwrap()), inp)
    }

    fn backward<S: Shape>(
        &self,
        op: super::PowiKernelOp,
        inp: &Self::Storage<S, E>,
        grad_inp: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        self.backward(
            super::PowfKernelOp(E::from_i32(op.0).unwrap()),
            inp,
            grad_inp,
            grad_out,
        )
    }
}
