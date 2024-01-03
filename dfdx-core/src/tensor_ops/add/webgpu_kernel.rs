use super::{BinaryAddKernelOp as Binary, ScalarAddKernelOp as Scalar};
use std::borrow::Cow;

use crate::prelude::{
    ops::{BinaryKernel, UnaryKernel},
    webgpu_kernels::webgpu_unary,
    Dtype, Webgpu,
};

const WGSL: &[u8] = b"TODO";

webgpu_unary!(Scalar<f32>, f32, WGSL, WGSL);

impl<E: Dtype> BinaryKernel<super::BinaryAddKernelOp, E> for Webgpu {
    const BACKWARD_WITHOUT_DATA: bool = true;

    fn forward<S: crate::prelude::Shape>(
        &self,
        op: super::BinaryAddKernelOp,
        lhs: Cow<crate::prelude::Tensor<S, E, Self>>,
        rhs: Cow<crate::prelude::Tensor<S, E, Self>>,
    ) -> Result<crate::prelude::Tensor<S, E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<S: crate::prelude::Shape>(
        &self,
        op: super::BinaryAddKernelOp,
        lhs: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
