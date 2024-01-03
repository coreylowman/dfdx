use std::borrow::Cow;

use crate::prelude::{ops::UnaryKernel, webgpu_kernels::webgpu_unary, Dtype, Webgpu};

const WGSL: &[u8] = b"TODO";

webgpu_unary!(super::PowfKernelOp<f32>, f32, WGSL, WGSL);

// TODO: Conflicting implementations of trait `UnaryKernel` for type `Webgpu`:
impl UnaryKernel<super::PowiKernelOp, f32> for Webgpu
where
    Self: UnaryKernel<super::PowfKernelOp<f32>, f32>,
{
    const BACKWARD_WITHOUT_INP: bool = false;

    const BACKWARD_WITHOUT_DATA: bool = false;

    fn forward<S: crate::prelude::Shape>(
        &self,
        op: super::PowiKernelOp,
        inp: Cow<crate::prelude::Tensor<S, f32, Self>>,
    ) -> Result<crate::prelude::Tensor<S, f32, Self>, crate::prelude::Error> {
        self.forward(super::PowfKernelOp(op.0 as f32), inp)
    }

    fn backward<S: crate::prelude::Shape>(
        &self,
        op: super::PowiKernelOp,
        inp: &impl crate::prelude::Tensorlike<S, f32, Self>,
        grad_inp: &mut Self::Vec,
        out: &impl crate::prelude::Tensorlike<S, f32, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        self.backward(
            super::PowfKernelOp(op.0 as f32),
            inp,
            grad_inp,
            out,
            grad_out,
        )
    }
}
