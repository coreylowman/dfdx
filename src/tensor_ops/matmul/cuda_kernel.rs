use crate::{
    shapes::*,
    tensor::{cuda::Cuda, Tensor},
};

use cudarc::{
    cublas::{
        result::CublasError, sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig,
        StridedBatchedConfig,
    },
    driver::{DevicePtr, DevicePtrMut},
};

const TRANS: cublasOperation_t = cublasOperation_t::CUBLAS_OP_T;
const NO_TRANS: cublasOperation_t = cublasOperation_t::CUBLAS_OP_N;

fn gemm_cfg<M: Dim, K: Dim, N: Dim, E: Dtype>(
    (m, k, n): (M, K, N),
    lhs_strides: [usize; 2],
    rhs_strides: [usize; 2],
    beta: E,
    out_strides: [usize; 2],
) -> (GemmConfig<E>, bool) {
    let (lhs_stride, lhs_trans) = super::matrix_strides((m.size(), k.size()), lhs_strides);
    let (rhs_stride, rhs_trans) = super::matrix_strides((k.size(), n.size()), rhs_strides);
    let (out_stride, out_trans) = super::matrix_strides((m.size(), n.size()), out_strides);

    if !out_trans {
        // out is stored in row major format
        let cfg = GemmConfig {
            transa: if rhs_trans { TRANS } else { NO_TRANS },
            transb: if lhs_trans { TRANS } else { NO_TRANS },
            m: n.size() as i32,
            n: m.size() as i32,
            k: k.size() as i32,
            alpha: E::ONE,
            lda: rhs_stride as i32,
            ldb: lhs_stride as i32,
            beta,
            ldc: out_stride as i32,
        };
        (cfg, true)
    } else {
        // out is stored in column major format
        let cfg = GemmConfig {
            transa: if lhs_trans { NO_TRANS } else { TRANS },
            transb: if rhs_trans { NO_TRANS } else { TRANS },
            m: m.size() as i32,
            n: n.size() as i32,
            k: k.size() as i32,
            alpha: E::ONE,
            lda: lhs_stride as i32,
            ldb: rhs_stride as i32,
            beta,
            ldc: out_stride as i32,
        };
        (cfg, false)
    }
}

impl Cuda {
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
    pub(crate) unsafe fn gemm<
        E: Dtype,
        M: Dim,
        K: Dim,
        N: Dim,
        A: DevicePtr<E>,
        B: DevicePtr<E>,
        C: DevicePtrMut<E>,
    >(
        &self,
        (m, k, n): (M, K, N),
        lhs: &A,
        lhs_strides: [usize; 2],
        rhs: &B,
        rhs_strides: [usize; 2],
        beta: E,
        out: &mut C,
        out_strides: [usize; 2],
    ) -> Result<(), CublasError>
    where
        CudaBlas: Gemm<E>,
    {
        let (cfg, swap_ops) = gemm_cfg((m, k, n), lhs_strides, rhs_strides, beta, out_strides);
        if !swap_ops {
            self.blas.gemm(cfg, lhs, rhs, out)
        } else {
            self.blas.gemm(cfg, rhs, lhs, out)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn gemm_batch<
        E: Dtype,
        Batch: Dim,
        M: Dim,
        K: Dim,
        N: Dim,
        A: DevicePtr<E>,
        B: DevicePtr<E>,
        C: DevicePtrMut<E>,
    >(
        &self,
        (batch, m, k, n): (Batch, M, K, N),
        lhs: &A,
        lhs_strides: [usize; 3],
        rhs: &B,
        rhs_strides: [usize; 3],
        beta: E,
        out: &mut C,
        out_strides: [usize; 3],
    ) -> Result<(), CublasError>
    where
        CudaBlas: Gemm<E>,
    {
        // NOTE: lhs_strides[0] and rhs_strides[0] can be 0
        assert_ne!(out_strides[0], 0);

        let (gemm, swap_ops) = gemm_cfg(
            (m, k, n),
            [lhs_strides[1], lhs_strides[2]],
            [rhs_strides[1], rhs_strides[2]],
            beta,
            [out_strides[1], out_strides[2]],
        );

        if !swap_ops {
            let cfg = StridedBatchedConfig {
                gemm,
                stride_a: lhs_strides[0] as i64,
                stride_b: rhs_strides[0] as i64,
                stride_c: out_strides[0] as i64,
                batch_size: batch.size() as i32,
            };
            self.blas.gemm_strided_batched(cfg, lhs, rhs, out)
        } else {
            let cfg = StridedBatchedConfig {
                gemm,
                stride_a: rhs_strides[0] as i64,
                stride_b: lhs_strides[0] as i64,
                stride_c: out_strides[0] as i64,
                batch_size: batch.size() as i32,
            };
            self.blas.gemm_strided_batched(cfg, rhs, lhs, out)
        }
    }
}

impl<E: Dtype> super::MatMatKernel<E> for Cuda
where
    CudaBlas: Gemm<E>,
{
    fn forward<M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(M, K), E, Self>,
        rhs: &Tensor<(K, N), E, Self>,
    ) -> Result<Tensor<(M, N), E, Self>, Self::Err> {
        let (m, _) = lhs.shape;
        let (k, n) = rhs.shape;
        let shape = (m, n);
        let strides = shape.strides();
        let mut storage = unsafe { self.alloc_empty::<E>(shape.num_elements()) }?;

        unsafe {
            self.gemm(
                (m, k, n),
                lhs.data.as_ref(),
                lhs.strides,
                rhs.data.as_ref(),
                rhs.strides,
                Default::default(),
                &mut storage,
                strides,
            )
        }?;

        Ok(self.build_tensor(shape, strides, storage))
    }

    fn backward<M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(M, K), E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<(K, N), E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let (m, _) = lhs.shape;
        let (k, n) = rhs.shape;
        let strides = (m, n).strides();
        self.par_stream.wait_for_default()?;
        unsafe {
            // grad_lhs += grad_out * rhs^T
            self.blas.set_stream(Some(self.par_stream.as_ref()))?;
            self.gemm(
                (m, n, k),
                grad_out,
                strides,
                rhs.data.as_ref(),
                [rhs.strides[1], rhs.strides[0]],
                E::ONE,
                grad_lhs,
                lhs.strides,
            )?;
            self.blas.set_stream(None)?;

            // grad_rhs += lhs^T * grad_out
            self.gemm(
                (k, m, n),
                lhs.data.as_ref(),
                [lhs.strides[1], lhs.strides[0]],
                grad_out,
                strides,
                E::ONE,
                grad_rhs,
                rhs.strides,
            )?;
        }
        self.dev.wait_for(self.par_stream.as_ref())?;
        Ok(())
    }
}

impl<E: Dtype> super::MatMatBrKernel<E> for Cuda
where
    CudaBlas: Gemm<E>,
{
    fn forward<B: Dim, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(B, M, K), E, Self>,
        rhs: &Tensor<(K, N), E, Self>,
    ) -> Result<Tensor<(B, M, N), E, Self>, Self::Err> {
        assert_ne!(lhs.strides[0], 0);
        let (batch, m, _) = lhs.shape;
        let (k, n) = rhs.shape;
        let shape = (batch, m, n);
        let strides = shape.strides();
        let mut storage = unsafe { self.alloc_empty::<E>(shape.num_elements()) }?;
        unsafe {
            self.gemm_batch(
                (batch, m, k, n),
                lhs.data.as_ref(),
                lhs.strides,
                rhs.data.as_ref(),
                [0, rhs.strides[0], rhs.strides[1]],
                Default::default(),
                &mut storage,
                strides,
            )?;
        }
        Ok(self.build_tensor(shape, strides, storage))
    }
    fn backward<B: Dim, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(B, M, K), E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<(K, N), E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let (batch, m, _) = lhs.shape;
        let (k, n) = rhs.shape;
        let strides = (batch, m, n).strides();
        self.par_stream.wait_for_default()?;
        unsafe {
            self.blas.set_stream(Some(self.par_stream.as_ref()))?;
            // grad_rhs += lhs^T * grad_out
            for i in 0..batch.size() {
                // NOTE: these have to be sequential since grad_rhs is broadcasted and cublas doesn't support
                // 0 stride
                self.gemm(
                    (k, m, n),
                    &lhs.data.slice(i * lhs.strides[0]..),
                    [lhs.strides[2], lhs.strides[1]],
                    &grad_out.slice(i * strides[0]..),
                    [strides[1], strides[2]],
                    E::ONE,
                    grad_rhs,
                    rhs.strides,
                )?;
            }
            self.blas.set_stream(None)?;

            // grad_lhs += grad_out * rhs^T
            self.gemm_batch(
                (batch, m, n, k),
                grad_out,
                strides,
                rhs.data.as_ref(),
                [0, rhs.strides[1], rhs.strides[0]],
                E::ONE,
                grad_lhs,
                lhs.strides,
            )?;
        }
        self.dev.wait_for(self.par_stream.as_ref())?;
        Ok(())
    }
}

impl<E: Dtype> super::MatMatBatch3Kernel<E> for Cuda
where
    CudaBlas: Gemm<E>,
{
    fn forward<B: Dim, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(B, M, K), E, Self>,
        rhs: &Tensor<(B, K, N), E, Self>,
    ) -> Result<Tensor<(B, M, N), E, Self>, Self::Err> {
        assert_ne!(lhs.strides[0], 0);
        assert_ne!(rhs.strides[0], 0);
        let (batch, m, _) = lhs.shape;
        let (_, k, n) = rhs.shape;
        let shape = (batch, m, n);
        let strides = shape.strides();
        let mut storage = unsafe { self.alloc_empty::<E>(shape.num_elements()) }?;
        unsafe {
            self.gemm_batch(
                (batch, m, k, n),
                lhs.data.as_ref(),
                lhs.strides,
                rhs.data.as_ref(),
                rhs.strides,
                Default::default(),
                &mut storage,
                strides,
            )?;
        }
        Ok(self.build_tensor(shape, strides, storage))
    }
    fn backward<B: Dim, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(B, M, K), E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<(B, K, N), E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let (batch, m, _) = lhs.shape;
        let (_, k, n) = rhs.shape;
        let strides = (batch, m, n).strides();
        self.par_stream.wait_for_default()?;
        unsafe {
            // grad_lhs += grad_out * rhs^T
            self.blas.set_stream(Some(self.par_stream.as_ref()))?;
            self.gemm_batch(
                (batch, m, n, k),
                grad_out,
                strides,
                rhs.data.as_ref(),
                [rhs.strides[0], rhs.strides[2], rhs.strides[1]],
                E::ONE,
                grad_lhs,
                lhs.strides,
            )?;
            self.blas.set_stream(None)?;

            // grad_rhs += lhs^T * grad_out
            self.gemm_batch(
                (batch, k, m, n),
                lhs.data.as_ref(),
                [lhs.strides[0], lhs.strides[2], lhs.strides[1]],
                grad_out,
                strides,
                E::ONE,
                grad_rhs,
                rhs.strides,
            )?;
        }
        self.dev.wait_for(self.par_stream.as_ref())?;
        Ok(())
    }
}

impl<E: Dtype> super::MatMatBatch4Kernel<E> for Cuda
where
    CudaBlas: Gemm<E>,
{
    fn forward<B: Dim, S: Dim, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(B, S, M, K), E, Self>,
        rhs: &Tensor<(B, S, K, N), E, Self>,
    ) -> Result<Tensor<(B, S, M, N), E, Self>, Self::Err> {
        assert_ne!(lhs.strides[0], 0);
        assert_ne!(rhs.strides[0], 0);
        assert_ne!(lhs.strides[1], 0);
        assert_ne!(rhs.strides[1], 0);

        let (batch, seq, m, _) = lhs.shape;
        let (_, _, k, n) = rhs.shape;
        let shape = (batch, seq, m, n);
        let strides = shape.strides();
        let mut storage = unsafe { self.alloc_empty::<E>(shape.num_elements()) }?;

        let half_batch = batch.size() / 2;

        self.par_stream.wait_for_default()?;

        unsafe {
            // split the batch onto separate streams
            self.blas.set_stream(Some(self.par_stream.as_ref()))?;
            for b in 0..half_batch {
                self.gemm_batch(
                    (seq, m, k, n),
                    &lhs.data.slice(b * lhs.strides[0]..),
                    [lhs.strides[1], lhs.strides[2], lhs.strides[3]],
                    &rhs.data.slice(b * rhs.strides[0]..),
                    [rhs.strides[1], rhs.strides[2], rhs.strides[3]],
                    Default::default(),
                    &mut storage.slice_mut(b * strides[0]..),
                    [strides[1], strides[2], strides[3]],
                )?;
            }
            self.blas.set_stream(None)?;

            for b in half_batch..batch.size() {
                self.gemm_batch(
                    (seq, m, k, n),
                    &lhs.data.slice(b * lhs.strides[0]..),
                    [lhs.strides[1], lhs.strides[2], lhs.strides[3]],
                    &rhs.data.slice(b * rhs.strides[0]..),
                    [rhs.strides[1], rhs.strides[2], rhs.strides[3]],
                    Default::default(),
                    &mut storage.slice_mut(b * strides[0]..),
                    [strides[1], strides[2], strides[3]],
                )?;
            }
        }
        self.dev.wait_for(self.par_stream.as_ref())?;
        Ok(self.build_tensor(shape, strides, storage))
    }

    fn backward<B: Dim, S: Dim, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(B, S, M, K), E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<(B, S, K, N), E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let (batch, seq, m, _) = lhs.shape;
        let (_, _, k, n) = rhs.shape;
        let strides = (batch, seq, m, n).strides();
        self.par_stream.wait_for_default()?;
        unsafe {
            // gl += go * rhs^T
            self.blas.set_stream(Some(self.par_stream.as_ref()))?;
            for i in 0..batch.size() {
                self.gemm_batch(
                    (seq, m, n, k),
                    &grad_out.slice(i * strides[0]..),
                    [strides[1], strides[2], strides[3]],
                    &rhs.data.slice(i * rhs.strides[0]..),
                    [rhs.strides[1], rhs.strides[3], rhs.strides[2]],
                    E::ONE,
                    &mut grad_lhs.slice_mut(i * lhs.strides[0]..),
                    [lhs.strides[1], lhs.strides[2], lhs.strides[3]],
                )?;
            }
            self.blas.set_stream(None)?;

            // gr += lhs^T * go
            for i in 0..batch.size() {
                self.gemm_batch(
                    (seq, k, m, n),
                    &lhs.data.slice(i * lhs.strides[0]..),
                    [lhs.strides[1], lhs.strides[3], lhs.strides[2]],
                    &grad_out.slice(i * strides[0]..),
                    [strides[1], strides[2], strides[3]],
                    E::ONE,
                    &mut grad_rhs.slice_mut(i * rhs.strides[0]..),
                    [rhs.strides[1], rhs.strides[2], rhs.strides[3]],
                )?;
            }
        }
        self.dev.wait_for(self.par_stream.as_ref())?;
        Ok(())
    }
}
