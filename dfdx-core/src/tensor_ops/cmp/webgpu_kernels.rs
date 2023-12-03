use crate::prelude::{Dtype, Webgpu};

impl<E: Dtype> super::CmpKernel<super::EqKernelOp, E> for Webgpu {
    fn forward<S: crate::prelude::Shape, T>(
        &self,
        lhs: &crate::prelude::Tensor<S, E, Self, T>,
        rhs: &crate::prelude::Tensor<S, E, Self, T>,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::CmpKernel<super::NeKernelOp, E> for Webgpu {
    fn forward<S: crate::prelude::Shape, T>(
        &self,
        lhs: &crate::prelude::Tensor<S, E, Self, T>,
        rhs: &crate::prelude::Tensor<S, E, Self, T>,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::CmpKernel<super::GtKernelOp, E> for Webgpu {
    fn forward<S: crate::prelude::Shape, T>(
        &self,
        lhs: &crate::prelude::Tensor<S, E, Self, T>,
        rhs: &crate::prelude::Tensor<S, E, Self, T>,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::CmpKernel<super::GeKernelOp, E> for Webgpu {
    fn forward<S: crate::prelude::Shape, T>(
        &self,
        lhs: &crate::prelude::Tensor<S, E, Self, T>,
        rhs: &crate::prelude::Tensor<S, E, Self, T>,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::CmpKernel<super::LtKernelOp, E> for Webgpu {
    fn forward<S: crate::prelude::Shape, T>(
        &self,
        lhs: &crate::prelude::Tensor<S, E, Self, T>,
        rhs: &crate::prelude::Tensor<S, E, Self, T>,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::CmpKernel<super::LeKernelOp, E> for Webgpu {
    fn forward<S: crate::prelude::Shape, T>(
        &self,
        lhs: &crate::prelude::Tensor<S, E, Self, T>,
        rhs: &crate::prelude::Tensor<S, E, Self, T>,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::ScalarCmpKernel<super::EqKernelOp, E> for Webgpu {
    fn forward<S: crate::prelude::Shape, T>(
        &self,
        lhs: &crate::prelude::Tensor<S, E, Self, T>,
        rhs: E,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::ScalarCmpKernel<super::NeKernelOp, E> for Webgpu {
    fn forward<S: crate::prelude::Shape, T>(
        &self,
        lhs: &crate::prelude::Tensor<S, E, Self, T>,
        rhs: E,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::ScalarCmpKernel<super::GtKernelOp, E> for Webgpu {
    fn forward<S: crate::prelude::Shape, T>(
        &self,
        lhs: &crate::prelude::Tensor<S, E, Self, T>,
        rhs: E,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::ScalarCmpKernel<super::GeKernelOp, E> for Webgpu {
    fn forward<S: crate::prelude::Shape, T>(
        &self,
        lhs: &crate::prelude::Tensor<S, E, Self, T>,
        rhs: E,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::ScalarCmpKernel<super::LtKernelOp, E> for Webgpu {
    fn forward<S: crate::prelude::Shape, T>(
        &self,
        lhs: &crate::prelude::Tensor<S, E, Self, T>,
        rhs: E,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::ScalarCmpKernel<super::LeKernelOp, E> for Webgpu {
    fn forward<S: crate::prelude::Shape, T>(
        &self,
        lhs: &crate::prelude::Tensor<S, E, Self, T>,
        rhs: E,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }
}
