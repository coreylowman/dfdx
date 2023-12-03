use crate::prelude::{Dtype, Webgpu};

impl<E: Dtype> super::MatMatKernel<E> for Webgpu {
    fn forward<M: crate::prelude::Dim, K: crate::prelude::Dim, N: crate::prelude::Dim>(
        &self,
        lhs: &crate::prelude::Tensor<(M, K), E, Self>,
        rhs: &crate::prelude::Tensor<(K, N), E, Self>,
    ) -> Result<crate::prelude::Tensor<(M, N), E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<M: crate::prelude::Dim, K: crate::prelude::Dim, N: crate::prelude::Dim>(
        &self,
        lhs: &crate::prelude::Tensor<(M, K), E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &crate::prelude::Tensor<(K, N), E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::MatMatBrKernel<E> for Webgpu {
    fn forward<
        B: crate::prelude::Dim,
        M: crate::prelude::Dim,
        K: crate::prelude::Dim,
        N: crate::prelude::Dim,
    >(
        &self,
        lhs: &crate::prelude::Tensor<(B, M, K), E, Self>,
        rhs: &crate::prelude::Tensor<(K, N), E, Self>,
    ) -> Result<crate::prelude::Tensor<(B, M, N), E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<
        B: crate::prelude::Dim,
        M: crate::prelude::Dim,
        K: crate::prelude::Dim,
        N: crate::prelude::Dim,
    >(
        &self,
        lhs: &crate::prelude::Tensor<(B, M, K), E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &crate::prelude::Tensor<(K, N), E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::MatMatBatch3Kernel<E> for Webgpu {
    fn forward<
        B: crate::prelude::Dim,
        M: crate::prelude::Dim,
        K: crate::prelude::Dim,
        N: crate::prelude::Dim,
    >(
        &self,
        lhs: &crate::prelude::Tensor<(B, M, K), E, Self>,
        rhs: &crate::prelude::Tensor<(B, K, N), E, Self>,
    ) -> Result<crate::prelude::Tensor<(B, M, N), E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<
        B: crate::prelude::Dim,
        M: crate::prelude::Dim,
        K: crate::prelude::Dim,
        N: crate::prelude::Dim,
    >(
        &self,
        lhs: &crate::prelude::Tensor<(B, M, K), E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &crate::prelude::Tensor<(B, K, N), E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}

impl<E: Dtype> super::MatMatBatch4Kernel<E> for Webgpu {
    fn forward<
        B: crate::prelude::Dim,
        S: crate::prelude::Dim,
        M: crate::prelude::Dim,
        K: crate::prelude::Dim,
        N: crate::prelude::Dim,
    >(
        &self,
        lhs: &crate::prelude::Tensor<(B, S, M, K), E, Self>,
        rhs: &crate::prelude::Tensor<(B, S, K, N), E, Self>,
    ) -> Result<crate::prelude::Tensor<(B, S, M, N), E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<
        B: crate::prelude::Dim,
        S: crate::prelude::Dim,
        M: crate::prelude::Dim,
        K: crate::prelude::Dim,
        N: crate::prelude::Dim,
    >(
        &self,
        lhs: &crate::prelude::Tensor<(B, S, M, K), E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &crate::prelude::Tensor<(B, S, K, N), E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
