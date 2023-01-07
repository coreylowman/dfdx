use crate::{
    shapes::*,
    tensor::cuda::{Cuda, CudaArray},
};

use cudarc::{
    blas::{Gemm, Gemv},
    cublas::sys::cublasOperation_t,
};
use std::sync::Arc;

impl super::VecVecKernel<f32> for Cuda {
    fn forward<M: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(M,), f32>,
        rhs: &Self::Storage<(N,), f32>,
    ) -> Result<Self::Storage<(M, N), f32>, Self::Err> {
        let shape = (lhs.shape.0, rhs.shape.0);
        let mut storage = self.dev.alloc_zeros_async::<f32>(shape.num_elements())?;

        // TODO: use strides
        let m = lhs.shape.0.size() as i32;
        let n = rhs.shape.0.size() as i32;
        let k = 1;
        unsafe {
            // storage = lhs * rhs
            self.blas.gemm_async(
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n,
                m,
                k,
                1.0,
                rhs.data.as_ref(),
                n,
                lhs.data.as_ref(),
                k,
                0.0,
                &mut storage,
                n,
            )
        }?;

        Ok(CudaArray {
            data: Arc::new(storage),
            shape,
            strides: shape.strides(),
        })
    }

    fn backward<M: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(M,), f32>,
        grad_lhs: &mut Self::Storage<(M,), f32>,
        rhs: &Self::Storage<(N,), f32>,
        grad_rhs: &mut Self::Storage<(N,), f32>,
        grad_out: &Self::Storage<(M, N), f32>,
    ) -> Result<(), Self::Err> {
        // TODO use strides
        {
            // grad_lhs += grad_out * rhs
            let m = lhs.shape.0.size() as i32;
            let n = 1;
            let k = rhs.shape.0.size() as i32;
            unsafe {
                self.blas.gemm_async(
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    1.0,
                    rhs.data.as_ref(),
                    n,
                    grad_out.data.as_ref(),
                    k,
                    1.0,
                    Arc::make_mut(&mut grad_lhs.data),
                    n,
                )
            }?;
        }
        {
            // grad_rhs += lhs * grad_out
            let m = 1;
            let n = rhs.shape.0.size() as i32;
            let k = lhs.shape.0.size() as i32;
            unsafe {
                self.blas.gemm_async(
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    1.0,
                    grad_out.data.as_ref(),
                    n,
                    lhs.data.as_ref(),
                    k,
                    1.0,
                    Arc::make_mut(&mut grad_rhs.data),
                    n,
                )
            }?;
        }
        Ok(())
    }
}

impl super::VecMatKernel<f32> for Cuda {
    fn forward<const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<K>,), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
    ) -> Result<Self::Storage<(N,), f32>, Self::Err> {
        todo!()
    }
    fn backward<const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<K>,), f32>,
        grad_lhs: &mut Self::Storage<(Const<K>,), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), f32>,
        grad_out: &Self::Storage<(N,), f32>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

impl super::MatMatKernel<f32> for Cuda {
    fn forward<M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
    ) -> Result<Self::Storage<(M, N), f32>, Self::Err> {
        todo!()
    }
    fn backward<M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(M, Const<K>), f32>,
        grad_lhs: &mut Self::Storage<(M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), f32>,
        grad_out: &Self::Storage<(M, N), f32>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

impl super::MatMatBrKernel<f32> for Cuda {
    fn forward<B: Dim, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(B, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
    ) -> Result<Self::Storage<(B, M, N), f32>, Self::Err> {
        todo!()
    }
    fn backward<B: Dim, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(B, M, Const<K>), f32>,
        grad_lhs: &mut Self::Storage<(B, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), f32>,
        grad_out: &Self::Storage<(B, M, N), f32>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

impl super::MatMatBatch3Kernel<f32> for Cuda {
    fn forward<const B: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<K>, N), f32>,
    ) -> Result<Self::Storage<(Const<B>, M, N), f32>, Self::Err> {
        todo!()
    }
    fn backward<const B: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, M, Const<K>), f32>,
        grad_lhs: &mut Self::Storage<(Const<B>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<B>, Const<K>, N), f32>,
        grad_out: &Self::Storage<(Const<B>, M, N), f32>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

impl super::MatMatBatch4Kernel<f32> for Cuda {
    fn forward<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
    ) -> Result<Self::Storage<(Const<B>, Const<S>, M, N), f32>, Self::Err> {
        todo!()
    }
    fn backward<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
        grad_lhs: &mut Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
        grad_out: &Self::Storage<(Const<B>, Const<S>, M, N), f32>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}
