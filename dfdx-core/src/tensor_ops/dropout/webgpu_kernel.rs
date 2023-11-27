use crate::prelude::{Dtype, Webgpu};

impl<E: Dtype> super::DropoutKernel<E> for Webgpu {
    fn forward<S: crate::prelude::Shape>(
        &self,
        op: super::DropoutKernelOp,
        inp: &crate::prelude::Tensor<S, E, Self>,
    ) -> Result<crate::prelude::Tensor<S, E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<S: crate::prelude::Shape>(
        &self,
        op: super::DropoutKernelOp,
        inp: &crate::prelude::Tensor<S, E, Self>,
        grad_inp: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
