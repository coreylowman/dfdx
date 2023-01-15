use crate::{
    shapes::*,
    tensor::cuda::{Cuda, CudaArray},
};

use cudarc::{
    cublas::{
        result::CublasError, sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig,
        StridedBatchedConfig,
    },
    driver::{DevicePtr, DevicePtrMut},
};
use std::sync::Arc;

/// sgemm helper.
///
/// # case 1: c is not transposed
/// note that lhs becomes b and rhs becomes a since cuda expects
/// column major data, but dfdx uses row major. So this function actually
/// calls the underlying cuda gemm call as `(N, K) * (K, M) = (N, M)`.
///
/// Since `(N, K)` in column major format is equal to (K, N) in row major format,
/// we can just pass rhs normally. We can pass the row major format `out` directly
/// in without transposing or anything because of this fact also.
///
/// # case 2: c is transposed
///
/// lhs is a and rhs is b, but we have to transpose them if they are not already
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm<
    M: Dim,
    K: Dim,
    N: Dim,
    A: DevicePtr<f32>,
    B: DevicePtr<f32>,
    C: DevicePtrMut<f32>,
>(
    blas: &CudaBlas,
    m: M,
    k: K,
    n: N,
    lhs: &A,
    lhs_strides: [usize; 2],
    rhs: &B,
    rhs_strides: [usize; 2],
    beta: f32,
    out: &mut C,
    out_strides: [usize; 2],
) -> Result<(), CublasError> {
    if out_strides[0] > 1 {
        // out is stored in row major format
        let (transa, lda) = if rhs_strides[0] > 1 {
            (cublasOperation_t::CUBLAS_OP_N, n.size() as i32)
        } else {
            (cublasOperation_t::CUBLAS_OP_T, k.size() as i32)
        };
        let (transb, ldb) = if lhs_strides[0] > 1 {
            (cublasOperation_t::CUBLAS_OP_N, k.size() as i32)
        } else {
            (cublasOperation_t::CUBLAS_OP_T, m.size() as i32)
        };
        let cfg = GemmConfig {
            transa,
            transb,
            m: n.size() as i32,
            n: m.size() as i32,
            k: k.size() as i32,
            alpha: 1.0,
            lda,
            ldb,
            beta,
            ldc: n.size() as i32,
        };
        blas.gemm_async(cfg, rhs, lhs, out)
    } else {
        // out is stored in column major format
        let (transa, lda) = if lhs_strides[0] > 1 {
            // lhs is stored in row major format
            (cublasOperation_t::CUBLAS_OP_T, k.size() as i32)
        } else {
            (cublasOperation_t::CUBLAS_OP_N, m.size() as i32)
        };
        let (transb, ldb) = if rhs_strides[0] > 1 {
            // rhs is stored in row major format
            (cublasOperation_t::CUBLAS_OP_T, n.size() as i32)
        } else {
            (cublasOperation_t::CUBLAS_OP_N, k.size() as i32)
        };

        // storage = lhs * rhs
        let cfg = GemmConfig {
            transa,
            transb,
            m: m.size() as i32,
            n: n.size() as i32,
            k: k.size() as i32,
            alpha: 1.0,
            lda,
            ldb,
            beta,
            ldc: m.size() as i32,
        };
        blas.gemm_async(cfg, lhs, rhs, out)
    }
}

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
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_op,
                n: m_op,
                k: k_op,
                alpha: 1.0,
                lda: n_op,
                ldb: k_op,
                beta: 0.0,
                ldc: n_op,
            };
            self.blas
                .gemm_async(cfg, rhs.data.as_ref(), lhs.data.as_ref(), &mut storage)
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
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_op,
                n: m_op,
                k: k_op,
                alpha: 1.0,
                lda: n_op,
                ldb: k_op,
                beta: 0.0,
                ldc: n_op,
            };
            unsafe {
                self.blas.gemm_async(
                    cfg,
                    rhs.data.as_ref(),
                    grad_out.data.as_ref(),
                    Arc::make_mut(&mut grad_lhs.data),
                )
            }?;
        }
        {
            // grad_rhs += lhs * grad_out
            let m_op = k.size() as i32;
            let n_op = n.size() as i32;
            let k_op = m.size() as i32;
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_op,
                n: m_op,
                k: k_op,
                alpha: 1.0,
                lda: n_op,
                ldb: k_op,
                beta: 1.0,
                ldc: n_op,
            };
            unsafe {
                self.blas.gemm_async(
                    cfg,
                    grad_out.data.as_ref(),
                    lhs.data.as_ref(),
                    Arc::make_mut(&mut grad_rhs.data),
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
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_op,
                n: m_op,
                k: k_op,
                alpha: 1.0,
                lda: n_op,
                ldb: k_op,
                beta: 0.0,
                ldc: n_op,
            };
            self.blas
                .gemm_async(cfg, rhs.data.as_ref(), lhs.data.as_ref(), &mut storage)?;
        }

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
        unsafe {
            // grad_lhs += grad_out * rhs^T
            let m_op = m.size() as i32;
            let n_op = k.size() as i32;
            let k_op = n.size() as i32;
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_op,
                n: m_op,
                k: k_op,
                alpha: 1.0,
                lda: k_op,
                ldb: k_op,
                beta: 1.0,
                ldc: n_op,
            };
            self.blas.gemm_async(
                cfg,
                rhs.data.as_ref(),
                grad_out.data.as_ref(),
                Arc::make_mut(&mut grad_lhs.data),
            )?;
        }
        unsafe {
            // grad_rhs += lhs * grad_out
            let m_op = k.size() as i32;
            let n_op = n.size() as i32;
            let k_op = m.size() as i32;
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_op,
                n: m_op,
                k: k_op,
                alpha: 1.0,
                lda: n_op,
                ldb: k_op,
                beta: 1.0,
                ldc: n_op,
            };
            self.blas.gemm_async(
                cfg,
                grad_out.data.as_ref(),
                lhs.data.as_ref(),
                Arc::make_mut(&mut grad_rhs.data),
            )?;
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
        let strides = shape.strides();
        let mut storage = self.dev.alloc_zeros_async::<f32>(shape.num_elements())?;

        unsafe {
            sgemm(
                self.blas.as_ref(),
                m,
                k,
                n,
                lhs.data.as_ref(),
                lhs.strides,
                rhs.data.as_ref(),
                rhs.strides,
                0.0,
                &mut storage,
                strides,
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
        unsafe {
            // grad_lhs += grad_out * rhs^T
            sgemm(
                self.blas.as_ref(),
                m,
                n,
                k,
                grad_out.data.as_ref(),
                grad_out.strides,
                rhs.data.as_ref(),
                [rhs.strides[1], rhs.strides[0]],
                1.0,
                Arc::make_mut(&mut grad_lhs.data),
                grad_lhs.strides,
            )?;
        }

        unsafe {
            // grad_rhs += lhs^T * grad_out
            sgemm(
                self.blas.as_ref(),
                k,
                m,
                n,
                lhs.data.as_ref(),
                [lhs.strides[1], lhs.strides[0]],
                grad_out.data.as_ref(),
                grad_out.strides,
                1.0,
                Arc::make_mut(&mut grad_rhs.data),
                grad_rhs.strides,
            )?;
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
        let (batch, m, _) = lhs.shape;
        let (k, n) = rhs.shape;
        let shape = (batch, m, n);
        let strides = shape.strides();
        let mut storage = self.dev.alloc_zeros_async::<f32>(shape.num_elements())?;

        // TODO: use strides
        unsafe {
            // storage = lhs * rhs
            let m_op = m.size() as i32;
            let n_op = n.size() as i32;
            let k_op = k.size() as i32;
            let cfg = StridedBatchedConfig {
                gemm: GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: n_op,
                    n: m_op,
                    k: k_op,
                    alpha: 1.0,
                    lda: n_op,
                    ldb: k_op,
                    beta: 0.0,
                    ldc: n_op,
                },
                stride_a: 0,
                stride_b: lhs.strides[0] as i64,
                stride_c: strides[0] as i64,
                batch_size: batch.size() as i32,
            };
            self.blas.gemm_strided_batched_async(
                cfg,
                rhs.data.as_ref(),
                lhs.data.as_ref(),
                &mut storage,
            )?;
        }
        Ok(CudaArray {
            data: Arc::new(storage),
            shape,
            strides,
        })
    }
    fn backward<B: Dim, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(B, M, Const<K>), f32>,
        grad_lhs: &mut Self::Storage<(B, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), f32>,
        grad_out: &Self::Storage<(B, M, N), f32>,
    ) -> Result<(), Self::Err> {
        let (batch, m, _) = lhs.shape;
        let (k, n) = rhs.shape;
        // TODO use strides
        unsafe {
            // grad_lhs += grad_out * rhs^T
            let m_op = m.size() as i32;
            let n_op = k.size() as i32;
            let k_op = n.size() as i32;
            let cfg = StridedBatchedConfig {
                gemm: GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: n_op,
                    n: m_op,
                    k: k_op,
                    alpha: 1.0,
                    lda: k_op,
                    ldb: k_op,
                    beta: 1.0,
                    ldc: n_op,
                },
                stride_a: 0,
                stride_b: grad_out.strides[0] as i64,
                stride_c: grad_lhs.strides[0] as i64,
                batch_size: batch.size() as i32,
            };
            self.blas.gemm_strided_batched_async(
                cfg,
                rhs.data.as_ref(),
                grad_out.data.as_ref(),
                Arc::make_mut(&mut grad_lhs.data),
            )?;
        }
        // TODO handle stride 0
        let grad_rhs_buf = Arc::make_mut(&mut grad_rhs.data);
        for b in 0..batch.size() {
            unsafe {
                // grad_rhs += lhs^T * grad_out
                let m_op = k.size() as i32;
                let n_op = n.size() as i32;
                let k_op = m.size() as i32;
                let cfg = GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: n_op,
                    n: m_op,
                    k: k_op,
                    alpha: 1.0,
                    lda: n_op,
                    ldb: m_op,
                    beta: 1.0,
                    ldc: n_op,
                };
                let grad_out_b = grad_out.data.try_slice(b * grad_out.strides[0]..).unwrap();
                let lhs_b = lhs.data.try_slice(b * lhs.strides[0]..).unwrap();
                self.blas
                    .gemm_async(cfg, &grad_out_b, &lhs_b, grad_rhs_buf)?;
            }
        }
        Ok(())
    }
}

impl super::MatMatBatch3Kernel<f32> for Cuda {
    fn forward<const B: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<K>, N), f32>,
    ) -> Result<Self::Storage<(Const<B>, M, N), f32>, Self::Err> {
        let (batch, m, _) = lhs.shape;
        let (_, k, n) = rhs.shape;
        let shape = (batch, m, n);
        let strides = shape.strides();
        let mut storage = self.dev.alloc_zeros_async::<f32>(shape.num_elements())?;

        // TODO: use strides
        unsafe {
            // storage = lhs * rhs
            let m_op = m.size() as i32;
            let n_op = n.size() as i32;
            let k_op = k.size() as i32;
            let cfg = StridedBatchedConfig {
                gemm: GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: n_op,
                    n: m_op,
                    k: k_op,
                    alpha: 1.0,
                    lda: n_op,
                    ldb: k_op,
                    beta: 0.0,
                    ldc: n_op,
                },
                stride_a: rhs.strides[0] as i64,
                stride_b: lhs.strides[0] as i64,
                stride_c: strides[0] as i64,
                batch_size: batch.size() as i32,
            };
            self.blas.gemm_strided_batched_async(
                cfg,
                rhs.data.as_ref(),
                lhs.data.as_ref(),
                &mut storage,
            )?;
        }
        Ok(CudaArray {
            data: Arc::new(storage),
            shape,
            strides,
        })
    }
    fn backward<const B: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, M, Const<K>), f32>,
        grad_lhs: &mut Self::Storage<(Const<B>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<B>, Const<K>, N), f32>,
        grad_out: &Self::Storage<(Const<B>, M, N), f32>,
    ) -> Result<(), Self::Err> {
        let (batch, m, _) = lhs.shape;
        let (_, k, n) = rhs.shape;
        // TODO use strides
        unsafe {
            // TODO handle strides[0] == 0
            // grad_lhs += grad_out * rhs^T
            let m_op = m.size() as i32;
            let n_op = k.size() as i32;
            let k_op = n.size() as i32;
            let cfg = StridedBatchedConfig {
                gemm: GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: n_op,
                    n: m_op,
                    k: k_op,
                    alpha: 1.0,
                    lda: k_op,
                    ldb: k_op,
                    beta: 1.0,
                    ldc: n_op,
                },
                stride_a: rhs.strides[0] as i64,
                stride_b: grad_out.strides[0] as i64,
                stride_c: grad_lhs.strides[0] as i64,
                batch_size: batch.size() as i32,
            };
            self.blas.gemm_strided_batched_async(
                cfg,
                rhs.data.as_ref(),
                grad_out.data.as_ref(),
                Arc::make_mut(&mut grad_lhs.data),
            )?;
        }
        unsafe {
            // TODO handle strides[0] == 0
            // grad_rhs += lhs^T * grad_out
            let m_op = k.size() as i32;
            let n_op = n.size() as i32;
            let k_op = m.size() as i32;
            let cfg = StridedBatchedConfig {
                gemm: GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: n_op,
                    n: m_op,
                    k: k_op,
                    alpha: 1.0,
                    lda: n_op,
                    ldb: m_op,
                    beta: 1.0,
                    ldc: n_op,
                },
                stride_a: grad_out.strides[0] as i64,
                stride_b: lhs.strides[0] as i64,
                stride_c: grad_rhs.strides[0] as i64,
                batch_size: batch.size() as i32,
            };
            self.blas.gemm_strided_batched_async(
                cfg,
                grad_out.data.as_ref(),
                lhs.data.as_ref(),
                Arc::make_mut(&mut grad_rhs.data),
            )?;
        }
        Ok(())
    }
}

impl super::MatMatBatch4Kernel<f32> for Cuda {
    fn forward<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
    ) -> Result<Self::Storage<(Const<B>, Const<S>, M, N), f32>, Self::Err> {
        let (batch, seq, m, _) = lhs.shape;
        let (_, _, k, n) = rhs.shape;
        let shape = (batch, seq, m, n);
        let strides = shape.strides();
        let mut storage = self.dev.alloc_zeros_async::<f32>(shape.num_elements())?;

        for b in 0..batch.size() {
            // TODO: use strides
            // TODO: use separate streams
            unsafe {
                // storage = lhs * rhs
                let m_op = m.size() as i32;
                let n_op = n.size() as i32;
                let k_op = k.size() as i32;
                let cfg = StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_N,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: n_op,
                        n: m_op,
                        k: k_op,
                        alpha: 1.0,
                        lda: n_op,
                        ldb: k_op,
                        beta: 0.0,
                        ldc: n_op,
                    },
                    stride_a: rhs.strides[1] as i64,
                    stride_b: lhs.strides[1] as i64,
                    stride_c: strides[1] as i64,
                    batch_size: seq.size() as i32,
                };
                let lhs_b = lhs.data.try_slice(b * lhs.strides[0]..).unwrap();
                let rhs_b = rhs.data.try_slice(b * rhs.strides[0]..).unwrap();
                let mut out_b = storage.try_slice_mut(b * strides[0]..).unwrap();
                self.blas
                    .gemm_strided_batched_async(cfg, &rhs_b, &lhs_b, &mut out_b)?;
            }
        }
        Ok(CudaArray {
            data: Arc::new(storage),
            shape,
            strides,
        })
    }

    fn backward<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
        grad_lhs: &mut Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
        grad_out: &Self::Storage<(Const<B>, Const<S>, M, N), f32>,
    ) -> Result<(), Self::Err> {
        let (batch, seq, m, _) = lhs.shape;
        let (_, _, k, n) = rhs.shape;
        // TODO use strides
        // TODO use streams
        let grad_lhs_buf = Arc::make_mut(&mut grad_lhs.data);
        for b in 0..batch.size() {
            unsafe {
                // TODO handle strides[1] == 0
                // grad_lhs += grad_out * rhs^T
                let m_op = m.size() as i32;
                let n_op = k.size() as i32;
                let k_op = n.size() as i32;
                let cfg = StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_T,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: n_op,
                        n: m_op,
                        k: k_op,
                        alpha: 1.0,
                        lda: k_op,
                        ldb: k_op,
                        beta: 1.0,
                        ldc: n_op,
                    },
                    stride_a: rhs.strides[1] as i64,
                    stride_b: grad_out.strides[1] as i64,
                    stride_c: grad_lhs.strides[1] as i64,
                    batch_size: seq.size() as i32,
                };
                let rhs_b = rhs.data.try_slice(b * rhs.strides[0]..).unwrap();
                let grad_out_b = grad_out.data.try_slice(b * grad_out.strides[0]..).unwrap();
                let mut grad_lhs_b = grad_lhs_buf
                    .try_slice_mut(b * grad_lhs.strides[0]..)
                    .unwrap();
                self.blas
                    .gemm_strided_batched_async(cfg, &rhs_b, &grad_out_b, &mut grad_lhs_b)?;
            }
        }
        let grad_rhs_buf = Arc::make_mut(&mut grad_rhs.data);
        for b in 0..batch.size() {
            unsafe {
                // TODO handle strides[1] == 0
                // grad_rhs += lhs^T * grad_out
                let m_op = k.size() as i32;
                let n_op = n.size() as i32;
                let k_op = m.size() as i32;
                let cfg = StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_N,
                        transb: cublasOperation_t::CUBLAS_OP_T,
                        m: n_op,
                        n: m_op,
                        k: k_op,
                        alpha: 1.0,
                        lda: n_op,
                        ldb: m_op,
                        beta: 1.0,
                        ldc: n_op,
                    },
                    stride_a: grad_out.strides[1] as i64,
                    stride_b: lhs.strides[1] as i64,
                    stride_c: grad_rhs.strides[1] as i64,
                    batch_size: seq.size() as i32,
                };
                let grad_out_b = grad_out.data.try_slice(b * grad_out.strides[0]..).unwrap();
                let lhs_b = lhs.data.try_slice(b * lhs.strides[0]..).unwrap();
                let mut grad_rhs_b = grad_rhs_buf
                    .try_slice_mut(b * grad_rhs.strides[0]..)
                    .unwrap();
                self.blas
                    .gemm_strided_batched_async(cfg, &grad_out_b, &lhs_b, &mut grad_rhs_b)?;
            }
        }
        Ok(())
    }
}
