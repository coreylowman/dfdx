use super::PowfKernelOp;
use crate::{
    dtypes::*,
    shapes::*,
    tensor::*,
    tensor_ops::{cuda_kernels::cuda_unary, ops::UnaryKernel},
};
use std::borrow::Cow;

#[cfg(feature = "f16")]
unsafe impl cudarc::driver::DeviceRepr for super::PowfKernelOp<f16> {}
#[cfg(feature = "f16")]
unsafe impl cudarc::driver::DeviceRepr for super::PowfKernelOp<AMP<f16>> {}
unsafe impl cudarc::driver::DeviceRepr for super::PowfKernelOp<f32> {}
unsafe impl cudarc::driver::DeviceRepr for super::PowfKernelOp<f64> {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/pow.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(PowfKernelOp<f16>, f16, PTX, "pow_fwd_f16", "pow_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(
    PowfKernelOp<AMP<f16>>,
    AMP<f16>,
    PTX,
    "pow_fwd_f16",
    "pow_bwd_f16"
);
cuda_unary!(PowfKernelOp<f32>, f32, PTX, "pow_fwd_f32", "pow_bwd_f32");
cuda_unary!(PowfKernelOp<f64>, f64, PTX, "pow_fwd_f64", "pow_bwd_f64");

impl<E: Dtype> UnaryKernel<super::PowiKernelOp, E> for Cuda
where
    Self: UnaryKernel<super::PowfKernelOp<E>, E>,
{
    const BACKWARD_WITHOUT_DATA: bool =
        <Self as UnaryKernel<super::PowfKernelOp<E>, E>>::BACKWARD_WITHOUT_DATA;
    const BACKWARD_WITHOUT_INP: bool =
        <Self as UnaryKernel<super::PowfKernelOp<E>, E>>::BACKWARD_WITHOUT_INP;
    fn forward<S: Shape>(
        &self,
        op: super::PowiKernelOp,
        inp: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.forward(super::PowfKernelOp(E::from_i32(op.0).unwrap()), inp)
    }

    fn backward<S: Shape>(
        &self,
        op: super::PowiKernelOp,
        inp: &impl Tensorlike<S, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &impl Tensorlike<S, E, Self>,
        grad_out: &Self::Vec,
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
