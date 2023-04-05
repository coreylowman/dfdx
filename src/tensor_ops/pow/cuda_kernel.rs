use super::PowfKernelOp;
use crate::{
    shapes::*,
    tensor::{Cuda, Tensor},
    tensor_ops::{cuda_kernels::cuda_unary, ops::UnaryKernel},
};

unsafe impl cudarc::driver::DeviceRepr for super::PowfKernelOp<f32> {}
unsafe impl cudarc::driver::DeviceRepr for super::PowfKernelOp<f64> {}

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
        inp: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.forward(super::PowfKernelOp(E::from_i32(op.0).unwrap()), inp)
    }

    fn backward<S: Shape>(
        &self,
        op: super::PowiKernelOp,
        inp: &Tensor<S, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<S, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        self.backward(
            super::PowfKernelOp(E::from_i32(op.0).unwrap()),
            inp,
            grad_inp,
            out,
            grad_out,
        )
    }
}
