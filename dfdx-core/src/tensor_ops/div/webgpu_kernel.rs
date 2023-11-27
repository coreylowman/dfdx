extern crate alloc;
use alloc::borrow::Cow;

use crate::prelude::{ops::{UnaryKernel, BinaryKernel}, Dtype, Webgpu};

impl<E: Dtype> UnaryKernel<super::ScalarDivKernelOp<E>, E> for Webgpu {
    const BACKWARD_WITHOUT_INP: bool = false;

    const BACKWARD_WITHOUT_DATA: bool = true;

    fn forward<S: crate::prelude::Shape>(
        &self,
        op: super::ScalarDivKernelOp<E>,
        inp: Cow<crate::prelude::Tensor<S, E, Self>>,
    ) -> Result<crate::prelude::Tensor<S, E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<S: crate::prelude::Shape>(
        &self,
        op: super::ScalarDivKernelOp<E>,
        inp: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}

impl <E:Dtype> BinaryKernel<super::BinaryDivKernelOp, E> for Webgpu {
    const BACKWARD_WITHOUT_DATA: bool = true;

    fn forward<S: crate::prelude::Shape>(
        &self,
        op: super::BinaryDivKernelOp,
        lhs: Cow<crate::prelude::Tensor<S, E, Self>>,
        rhs: Cow<crate::prelude::Tensor<S, E, Self>>,
    ) -> Result<crate::prelude::Tensor<S, E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<S: crate::prelude::Shape>(
        &self,
        op: super::BinaryDivKernelOp,
        lhs: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
