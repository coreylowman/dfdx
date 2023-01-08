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
        let (m,) = lhs.shape;
        let (n,) = rhs.shape;
        let k = Const::<1>;
        let shape = (m, n);
        let mut storage = self.dev.alloc_zeros_async::<f32>(shape.num_elements())?;

        // TODO: use strides
        unsafe {
            // storage = lhs * rhs
            let m_op = m.size() as i32;
            let n_op = n.size() as i32;
            let k_op = k.size() as i32;
            self.blas.gemm_async(
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n_op,
                m_op,
                k_op,
                1.0,
                rhs.data.as_ref(),
                n_op,
                lhs.data.as_ref(),
                k_op,
                0.0,
                &mut storage,
                n_op,
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
        let (m, n) = grad_out.shape;
        let k = Const::<1>;
        // TODO use strides
        {
            // grad_lhs += grad_out * rhs
            let m_op = m.size() as i32;
            let n_op = k.size() as i32;
            let k_op = n.size() as i32;
            unsafe {
                self.blas.gemm_async(
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    n_op,
                    m_op,
                    k_op,
                    1.0,
                    rhs.data.as_ref(),
                    n_op,
                    grad_out.data.as_ref(),
                    k_op,
                    1.0,
                    Arc::make_mut(&mut grad_lhs.data),
                    n_op,
                )
            }?;
        }
        {
            // grad_rhs += lhs * grad_out
            let m_op = k.size() as i32;
            let n_op = n.size() as i32;
            let k_op = m.size() as i32;
            unsafe {
                self.blas.gemm_async(
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    n_op,
                    m_op,
                    k_op,
                    1.0,
                    grad_out.data.as_ref(),
                    n_op,
                    lhs.data.as_ref(),
                    k_op,
                    1.0,
                    Arc::make_mut(&mut grad_rhs.data),
                    n_op,
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
        let m = Const::<1>;
        let (k, n) = rhs.shape;
        let shape = (n,);
        let mut storage = self.dev.alloc_zeros_async::<f32>(shape.num_elements())?;

        // TODO: use strides
        unsafe {
            // storage = lhs * rhs
            let m_op = m.size() as i32;
            let n_op = n.size() as i32;
            let k_op = k.size() as i32;
            self.blas.gemm_async(
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n_op,
                m_op,
                k_op,
                1.0,
                rhs.data.as_ref(),
                n_op,
                lhs.data.as_ref(),
                k_op,
                0.0,
                &mut storage,
                n_op,
            )
        }?;

        Ok(CudaArray {
            data: Arc::new(storage),
            shape,
            strides: shape.strides(),
        })
    }
    fn backward<const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<K>,), f32>,
        grad_lhs: &mut Self::Storage<(Const<K>,), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), f32>,
        grad_out: &Self::Storage<(N,), f32>,
    ) -> Result<(), Self::Err> {
        let m = Const::<1>;
        let (k, n) = rhs.shape;
        // TODO use strides
        {
            // grad_lhs += grad_out * rhs^T
            let m_op = m.size() as i32;
            let n_op = k.size() as i32;
            let k_op = n.size() as i32;
            unsafe {
                self.blas.gemm_async(
                    cublasOperation_t::CUBLAS_OP_T,
                    cublasOperation_t::CUBLAS_OP_N,
                    n_op,
                    m_op,
                    k_op,
                    1.0,
                    rhs.data.as_ref(),
                    k_op,
                    grad_out.data.as_ref(),
                    k_op,
                    1.0,
                    Arc::make_mut(&mut grad_lhs.data),
                    n_op,
                )
            }?;
        }
        {
            // grad_rhs += lhs * grad_out
            let m_op = k.size() as i32;
            let n_op = n.size() as i32;
            let k_op = m.size() as i32;
            unsafe {
                self.blas.gemm_async(
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    n_op,
                    m_op,
                    k_op,
                    1.0,
                    grad_out.data.as_ref(),
                    n_op,
                    lhs.data.as_ref(),
                    k_op,
                    1.0,
                    Arc::make_mut(&mut grad_rhs.data),
                    n_op,
                )
            }?;
        }
        Ok(())
    }
}

impl super::MatMatKernel<f32> for Cuda {
    fn forward<M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
    ) -> Result<Self::Storage<(M, N), f32>, Self::Err> {
        let (m, _) = lhs.shape;
        let (k, n) = rhs.shape;
        let shape = (m, n);
        let mut storage = self.dev.alloc_zeros_async::<f32>(shape.num_elements())?;

        // TODO: use strides
        unsafe {
            // storage = lhs * rhs
            let m_op = m.size() as i32;
            let n_op = n.size() as i32;
            let k_op = k.size() as i32;
            self.blas.gemm_async(
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n_op,
                m_op,
                k_op,
                1.0,
                rhs.data.as_ref(),
                n_op,
                lhs.data.as_ref(),
                k_op,
                0.0,
                &mut storage,
                n_op,
            )
        }?;

        Ok(CudaArray {
            data: Arc::new(storage),
            shape,
            strides: shape.strides(),
        })
    }
    fn backward<M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(M, Const<K>), f32>,
        grad_lhs: &mut Self::Storage<(M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), f32>,
        grad_out: &Self::Storage<(M, N), f32>,
    ) -> Result<(), Self::Err> {
        let (m, _) = lhs.shape;
        let (k, n) = rhs.shape;
        // TODO use strides
        {
            // grad_lhs += grad_out * rhs^T
            let m_op = m.size() as i32;
            let n_op = k.size() as i32;
            let k_op = n.size() as i32;
            unsafe {
                self.blas.gemm_async(
                    cublasOperation_t::CUBLAS_OP_T,
                    cublasOperation_t::CUBLAS_OP_N,
                    n_op,
                    m_op,
                    k_op,
                    1.0,
                    rhs.data.as_ref(),
                    k_op,
                    grad_out.data.as_ref(),
                    k_op,
                    1.0,
                    Arc::make_mut(&mut grad_lhs.data),
                    n_op,
                )
            }?;
        }
        {
            // grad_rhs += lhs^T * grad_out
            let m_op = k.size() as i32;
            let n_op = n.size() as i32;
            let k_op = m.size() as i32;
            unsafe {
                self.blas.gemm_async(
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_T,
                    n_op,
                    m_op,
                    k_op,
                    1.0,
                    grad_out.data.as_ref(),
                    n_op,
                    lhs.data.as_ref(),
                    m_op,
                    1.0,
                    Arc::make_mut(&mut grad_rhs.data),
                    n_op,
                )
            }?;
        }
        Ok(())
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
