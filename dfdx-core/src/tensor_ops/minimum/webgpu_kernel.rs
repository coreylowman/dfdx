use crate::prelude::{ops::BinaryKernel, Dtype, Webgpu};
use std::borrow::Cow;

impl<E: Dtype> BinaryKernel<super::MinimumKernelOp, E> for Webgpu {
    const BACKWARD_WITHOUT_DATA: bool = false;

    fn forward<S: crate::prelude::Shape>(
        &self,
        op: super::MinimumKernelOp,
        lhs: Cow<crate::prelude::Tensor<S, E, Self>>,
        rhs: Cow<crate::prelude::Tensor<S, E, Self>>,
    ) -> Result<crate::prelude::Tensor<S, E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<S: crate::prelude::Shape>(
        &self,
        op: super::MinimumKernelOp,
        lhs: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
