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

const TRANS: cublasOperation_t = cublasOperation_t::CUBLAS_OP_T;
const NO_TRANS: cublasOperation_t = cublasOperation_t::CUBLAS_OP_N;

fn sgemm_config<M: Dim, K: Dim, N: Dim>(
    (m, k, n): (M, K, N),
    lhs_strides: [usize; 2],
    rhs_strides: [usize; 2],
    beta: f32,
    out_strides: [usize; 2],
) -> GemmConfig<f32> {
    let (lhs_stride, lhs_trans) = match lhs_strides {
        [1, 0] => (m.size(), true),
        [0, 1] => (k.size(), false),
        [ld, 1] => (ld, false),
        [1, ld] => (ld, true),
        _ => panic!("At least one of a's strides must be 1 for cublas"),
    };

    let (rhs_stride, rhs_trans) = match rhs_strides {
        [1, 0] => (k.size(), true),
        [0, 1] => (n.size(), false),
        [ld, 1] => (ld, false),
        [1, ld] => (ld, true),
        _ => panic!("At least one of b's strides must be 1 for cublas"),
    };

    let (out_stride, out_trans) = match out_strides {
        [1, 0] => (m.size(), true),
        [0, 1] => (n.size(), false),
        [ld, 1] => (ld, false),
        [1, ld] => (ld, true),
        _ => panic!("At least one of c's strides must be 1 for cublas"),
    };

    if !out_trans {
        // out is stored in row major format
        GemmConfig {
            transa: if rhs_trans { TRANS } else { NO_TRANS },
            transb: if lhs_trans { TRANS } else { NO_TRANS },
            m: n.size() as i32,
            n: m.size() as i32,
            k: k.size() as i32,
            alpha: 1.0,
            lda: rhs_stride as i32,
            ldb: lhs_stride as i32,
            beta,
            ldc: out_stride as i32,
        }
    } else {
        // out is stored in column major format
        GemmConfig {
            transa: if lhs_trans { NO_TRANS } else { TRANS },
            transb: if rhs_trans { NO_TRANS } else { TRANS },
            m: m.size() as i32,
            n: n.size() as i32,
            k: k.size() as i32,
            alpha: 1.0,
            lda: lhs_stride as i32,
            ldb: rhs_stride as i32,
            beta,
            ldc: out_stride as i32,
        }
    }
}

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
pub(crate) unsafe fn sgemm<
    M: Dim,
    K: Dim,
    N: Dim,
    A: DevicePtr<f32>,
    B: DevicePtr<f32>,
    C: DevicePtrMut<f32>,
>(
    blas: &CudaBlas,
    (m, k, n): (M, K, N),
    lhs: &A,
    lhs_strides: [usize; 2],
    rhs: &B,
    rhs_strides: [usize; 2],
    beta: f32,
    out: &mut C,
    out_strides: [usize; 2],
) -> Result<(), CublasError> {
    let cfg = sgemm_config((m, k, n), lhs_strides, rhs_strides, beta, out_strides);

    if cfg.m == m.size() as i32 {
        blas.gemm_async(cfg, lhs, rhs, out)
    } else {
        debug_assert_eq!(cfg.m, n.size() as i32);
        blas.gemm_async(cfg, rhs, lhs, out)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn sgemm_batch<
    Batch: Dim,
    M: Dim,
    K: Dim,
    N: Dim,
    A: DevicePtr<f32>,
    B: DevicePtr<f32>,
    C: DevicePtrMut<f32>,
>(
    blas: &CudaBlas,
    (batch, m, k, n): (Batch, M, K, N),
    lhs: &A,
    lhs_strides: [usize; 3],
    rhs: &B,
    rhs_strides: [usize; 3],
    beta: f32,
    out: &mut C,
    out_strides: [usize; 3],
) -> Result<(), CublasError> {
    // NOTE: lhs_strides[0] and rhs_strides[0] can be 0
    assert_ne!(out_strides[0], 0);

    let gemm = sgemm_config(
        (m, k, n),
        [lhs_strides[1], lhs_strides[2]],
        [rhs_strides[1], rhs_strides[2]],
        beta,
        [out_strides[1], out_strides[2]],
    );

    if gemm.m == m.size() as i32 {
        let cfg = StridedBatchedConfig {
            gemm,
            stride_a: lhs_strides[0] as i64,
            stride_b: rhs_strides[0] as i64,
            stride_c: out_strides[0] as i64,
            batch_size: batch.size() as i32,
        };
        blas.gemm_strided_batched_async(cfg, lhs, rhs, out)
    } else {
        debug_assert_eq!(gemm.m, n.size() as i32);
        let cfg = StridedBatchedConfig {
            gemm,
            stride_a: rhs_strides[0] as i64,
            stride_b: lhs_strides[0] as i64,
            stride_c: out_strides[0] as i64,
            batch_size: batch.size() as i32,
        };
        blas.gemm_strided_batched_async(cfg, rhs, lhs, out)
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
        let strides = shape.strides();
        let mut storage = self.dev.alloc_zeros_async::<f32>(shape.num_elements())?;

        unsafe {
            sgemm(
                self.blas.as_ref(),
                (m, k, n),
                lhs.data.as_ref(),
                [lhs.strides[0], 0],
                rhs.data.as_ref(),
                [0, rhs.strides[0]],
                0.0,
                &mut storage,
                strides,
            )?;
        }

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
        unsafe {
            // grad_lhs += grad_out * rhs^T
            sgemm(
                self.blas.as_ref(),
                (m, n, k),
                grad_out.data.as_ref(),
                grad_out.strides,
                rhs.data.as_ref(),
                [rhs.strides[0], 0],
                1.0,
                Arc::make_mut(&mut grad_lhs.data),
                [grad_lhs.strides[0], 0],
            )?;
        }
        unsafe {
            // grad_rhs += lhs^T * grad_out
            sgemm(
                self.blas.as_ref(),
                (k, m, n),
                lhs.data.as_ref(),
                [0, lhs.strides[0]],
                grad_out.data.as_ref(),
                grad_out.strides,
                1.0,
                Arc::make_mut(&mut grad_rhs.data),
                [0, grad_rhs.strides[0]],
            )?;
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
        let strides = shape.strides();
        let mut storage = self.dev.alloc_zeros_async::<f32>(shape.num_elements())?;

        unsafe {
            sgemm(
                self.blas.as_ref(),
                (m, k, n),
                lhs.data.as_ref(),
                [0, lhs.strides[0]],
                rhs.data.as_ref(),
                rhs.strides,
                0.0,
                &mut storage,
                [0, strides[0]],
            )?;
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
        unsafe {
            // grad_lhs += grad_out * rhs^T
            sgemm(
                self.blas.as_ref(),
                (m, n, k),
                grad_out.data.as_ref(),
                [0, grad_out.strides[0]],
                rhs.data.as_ref(),
                [rhs.strides[1], rhs.strides[0]],
                1.0,
                Arc::make_mut(&mut grad_lhs.data),
                [0, grad_lhs.strides[0]],
            )?;

            // grad_rhs += lhs^T * grad_out
            sgemm(
                self.blas.as_ref(),
                (k, m, n),
                lhs.data.as_ref(),
                [lhs.strides[0], 0],
                grad_out.data.as_ref(),
                [0, grad_out.strides[0]],
                1.0,
                Arc::make_mut(&mut grad_rhs.data),
                grad_rhs.strides,
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
                (m, k, n),
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
                (m, n, k),
                grad_out.data.as_ref(),
                grad_out.strides,
                rhs.data.as_ref(),
                [rhs.strides[1], rhs.strides[0]],
                1.0,
                Arc::make_mut(&mut grad_lhs.data),
                grad_lhs.strides,
            )?;

            // grad_rhs += lhs^T * grad_out
            sgemm(
                self.blas.as_ref(),
                (k, m, n),
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

        unsafe {
            sgemm_batch(
                self.blas.as_ref(),
                (batch, m, k, n),
                lhs.data.as_ref(),
                lhs.strides,
                rhs.data.as_ref(),
                [0, rhs.strides[0], rhs.strides[1]],
                0.0,
                &mut storage,
                strides,
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
        assert_ne!(grad_lhs.strides[0], 0);
        let (batch, m, _) = lhs.shape;
        let (k, n) = rhs.shape;
        unsafe {
            // grad_lhs += grad_out * rhs^T
            sgemm_batch(
                self.blas.as_ref(),
                (batch, m, n, k),
                grad_out.data.as_ref(),
                grad_out.strides,
                rhs.data.as_ref(),
                [0, rhs.strides[1], rhs.strides[0]],
                1.0,
                Arc::make_mut(&mut grad_lhs.data),
                grad_lhs.strides,
            )?;
        }
        let grad_rhs_buf = Arc::make_mut(&mut grad_rhs.data);
        for b in 0..batch.size() {
            // NOTE: these have to be sequential since grad_rhs is broadcasted and cublas doesn't support
            // 0 strides with atomicAdd
            unsafe {
                // grad_rhs += lhs^T * grad_out
                sgemm(
                    self.blas.as_ref(),
                    (k, m, n),
                    &lhs.data.try_slice(b * lhs.strides[0]..).unwrap(),
                    [lhs.strides[2], lhs.strides[1]],
                    &grad_out.data.try_slice(b * grad_out.strides[0]..).unwrap(),
                    [grad_out.strides[1], grad_out.strides[2]],
                    1.0,
                    grad_rhs_buf,
                    grad_rhs.strides,
                )?;
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

        unsafe {
            sgemm_batch(
                self.blas.as_ref(),
                (batch, m, k, n),
                lhs.data.as_ref(),
                lhs.strides,
                rhs.data.as_ref(),
                rhs.strides,
                0.0,
                &mut storage,
                strides,
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
        assert_ne!(grad_lhs.strides[0], 0);
        assert_ne!(grad_rhs.strides[0], 0);
        let (batch, m, _) = lhs.shape;
        let (_, k, n) = rhs.shape;
        unsafe {
            // grad_lhs += grad_out * rhs^T
            sgemm_batch(
                self.blas.as_ref(),
                (batch, m, n, k),
                grad_out.data.as_ref(),
                grad_out.strides,
                rhs.data.as_ref(),
                [rhs.strides[0], rhs.strides[2], rhs.strides[1]],
                1.0,
                Arc::make_mut(&mut grad_lhs.data),
                grad_lhs.strides,
            )?;

            // grad_rhs += lhs^T * grad_out
            sgemm_batch(
                self.blas.as_ref(),
                (batch, k, m, n),
                lhs.data.as_ref(),
                [lhs.strides[0], lhs.strides[2], lhs.strides[1]],
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
            // TODO: use separate streams
            unsafe {
                sgemm_batch(
                    self.blas.as_ref(),
                    (seq, m, k, n),
                    &lhs.data.try_slice(b * lhs.strides[0]..).unwrap(),
                    [lhs.strides[1], lhs.strides[2], lhs.strides[3]],
                    &rhs.data.try_slice(b * rhs.strides[0]..).unwrap(),
                    [rhs.strides[1], rhs.strides[2], rhs.strides[3]],
                    0.0,
                    &mut storage.try_slice_mut(b * strides[0]..).unwrap(),
                    [strides[1], strides[2], strides[3]],
                )?;
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
        gl: &mut Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
        gr: &mut Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
        go: &Self::Storage<(Const<B>, Const<S>, M, N), f32>,
    ) -> Result<(), Self::Err> {
        assert_ne!(gl.strides[0], 0);
        assert_ne!(gr.strides[0], 0);
        assert_ne!(gl.strides[1], 0);
        assert_ne!(gr.strides[1], 0);

        let (batch, seq, m, _) = lhs.shape;
        let (_, _, k, n) = rhs.shape;
        // TODO use streams
        let gl_buf = Arc::make_mut(&mut gl.data);
        let gr_buf = Arc::make_mut(&mut gr.data);

        for b in 0..batch.size() {
            unsafe {
                // gl += go * rhs^T
                sgemm_batch(
                    self.blas.as_ref(),
                    (seq, m, n, k),
                    &go.data.try_slice(b * go.strides[0]..).unwrap(),
                    [go.strides[1], go.strides[2], go.strides[3]],
                    &rhs.data.try_slice(b * rhs.strides[0]..).unwrap(),
                    [rhs.strides[1], rhs.strides[3], rhs.strides[2]],
                    1.0,
                    &mut gl_buf.try_slice_mut(b * gl.strides[0]..).unwrap(),
                    [gl.strides[1], gl.strides[2], gl.strides[3]],
                )?;

                // gr += lhs^T * go
                sgemm_batch(
                    self.blas.as_ref(),
                    (seq, k, m, n),
                    &lhs.data.try_slice(b * lhs.strides[0]..).unwrap(),
                    [lhs.strides[1], lhs.strides[3], lhs.strides[2]],
                    &go.data.try_slice(b * go.strides[0]..).unwrap(),
                    [go.strides[1], go.strides[2], go.strides[3]],
                    1.0,
                    &mut gr_buf.try_slice_mut(b * gr.strides[0]..).unwrap(),
                    [gr.strides[1], gr.strides[2], gr.strides[3]],
                )?;
            }
        }
        Ok(())
    }
}
