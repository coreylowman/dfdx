use std::borrow::Cow;

use crate::prelude::{ops::UnaryKernel, Dtype, Webgpu};

impl<E: Dtype> UnaryKernel<super::PowfKernelOp<E>, E> for Webgpu {
    const BACKWARD_WITHOUT_INP: bool = false;

    const BACKWARD_WITHOUT_DATA: bool = false;

    fn forward<S: crate::prelude::Shape>(
        &self,
        op: super::PowfKernelOp<E>,
        inp: Cow<crate::prelude::Tensor<S, E, Self>>,
    ) -> Result<crate::prelude::Tensor<S, E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<S: crate::prelude::Shape>(
        &self,
        op: super::PowfKernelOp<E>,
        inp: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> UnaryKernel<super::PowiKernelOp, E> for Webgpu {
    const BACKWARD_WITHOUT_INP: bool = false;

    const BACKWARD_WITHOUT_DATA: bool = false;

    fn forward<S: crate::prelude::Shape>(
        &self,
        op: super::PowiKernelOp,
        inp: Cow<crate::prelude::Tensor<S, E, Self>>,
    ) -> Result<crate::prelude::Tensor<S, E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<S: crate::prelude::Shape>(
        &self,
        op: super::PowiKernelOp,
        inp: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
