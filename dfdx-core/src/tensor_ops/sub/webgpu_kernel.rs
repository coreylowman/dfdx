use std::borrow::Cow;

use super::{BinarySubKernelOp as Binary, ScalarSubKernelOp as Scalar};

use crate::prelude::{ops::BinaryKernel, webgpu_kernels::webgpu_unary, Dtype, Webgpu};

const WGSL: &str = "TODO";

webgpu_unary!(const_df() Scalar<f32>, f32, WGSL, "scalar_sub_fwd", "scalar_sub_bwd");

impl<E: Dtype> BinaryKernel<super::BinarySubKernelOp, E> for Webgpu {
    const BACKWARD_WITHOUT_DATA: bool = true;

    fn forward<S: crate::prelude::Shape>(
        &self,
        op: super::BinarySubKernelOp,
        lhs: Cow<crate::prelude::Tensor<S, E, Self>>,
        rhs: Cow<crate::prelude::Tensor<S, E, Self>>,
    ) -> Result<crate::prelude::Tensor<S, E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<S: crate::prelude::Shape>(
        &self,
        op: super::BinarySubKernelOp,
        lhs: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
