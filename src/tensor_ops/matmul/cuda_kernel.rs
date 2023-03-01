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

fn sgemm_config<M: Dim, K: Dim, N: Dim, E: Dtype>(
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
    E: Dtype,
    M: Dim,
    K: Dim,
    N: Dim,
    A: DevicePtr<E>,
    B: DevicePtr<E>,
    C: DevicePtrMut<E>,
>(
    blas: &CudaBlas,
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
    let (cfg, swap_ops) = sgemm_config((m, k, n), lhs_strides, rhs_strides, beta, out_strides);

    if !swap_ops {
        blas.gemm_async(cfg, lhs, rhs, out)
    } else {
        blas.gemm_async(cfg, rhs, lhs, out)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn sgemm_batch<
    E: Dtype,
    Batch: Dim,
    M: Dim,
    K: Dim,
    N: Dim,
    A: DevicePtr<E>,
    B: DevicePtr<E>,
    C: DevicePtrMut<E>,
>(
    blas: &CudaBlas,
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

    let (gemm, swap_ops) = sgemm_config(
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
        blas.gemm_strided_batched_async(cfg, lhs, rhs, out)
    } else {
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

impl<E: Dtype> super::VecVecKernel<E> for Cuda
where
    CudaBlas: Gemm<E>,
{
    fn forward<M: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(M,), E, Self>,
        rhs: &Tensor<(N,), E, Self>,
    ) -> Result<Tensor<(M, N), E, Self>, Self::Err> {
        let (m,) = lhs.shape;
        let (n,) = rhs.shape;
        let k = Const::<1>;
        let shape = (m, n);
        let strides = shape.strides();
        let mut storage = unsafe { self.dev.alloc_async::<E>(shape.num_elements()) }?;

        unsafe {
            sgemm(
                self.blas.as_ref(),
                (m, k, n),
                lhs.data.as_ref(),
                [lhs.strides[0], 0],
                rhs.data.as_ref(),
                [0, rhs.strides[0]],
                Default::default(),
                &mut storage,
                strides,
            )?;
        }

        Ok(self.build_tensor(shape, shape.strides(), storage))
    }

    fn backward<M: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(M,), E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<(N,), E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let m = lhs.shape.0;
        let k = Const::<1>;
        let n = rhs.shape.0;
        let strides = (m, n).strides();
        unsafe {
            // grad_lhs += grad_out * rhs^T
            sgemm(
                self.blas.as_ref(),
                (m, n, k),
                grad_out,
                strides,
                rhs.data.as_ref(),
                [rhs.strides[0], 0],
                E::ONE,
                grad_lhs,
                [lhs.strides[0], 0],
            )?;
        }
        unsafe {
            // grad_rhs += lhs^T * grad_out
            sgemm(
                self.blas.as_ref(),
                (k, m, n),
                lhs.data.as_ref(),
                [0, lhs.strides[0]],
                grad_out,
                strides,
                E::ONE,
                grad_rhs,
                [0, rhs.strides[0]],
            )?;
        }
        Ok(())
    }
}

impl<E: Dtype> super::VecMatKernel<E> for Cuda
where
    CudaBlas: Gemm<E>,
{
    fn forward<K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(K,), E, Self>,
        rhs: &Tensor<(K, N), E, Self>,
    ) -> Result<Tensor<(N,), E, Self>, Self::Err> {
        let m = Const::<1>;
        let (k, n) = rhs.shape;
        let shape = (n,);
        let strides = shape.strides();
        let mut storage = unsafe { self.dev.alloc_async::<E>(shape.num_elements()) }?;

        unsafe {
            sgemm(
                self.blas.as_ref(),
                (m, k, n),
                lhs.data.as_ref(),
                [0, lhs.strides[0]],
                rhs.data.as_ref(),
                rhs.strides,
                Default::default(),
                &mut storage,
                [0, strides[0]],
            )?;
        }

        Ok(self.build_tensor(shape, shape.strides(), storage))
    }
    fn backward<K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(K,), E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<(K, N), E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let m = Const::<1>;
        let (k, n) = rhs.shape;
        unsafe {
            // grad_lhs += grad_out * rhs^T
            sgemm(
                self.blas.as_ref(),
                (m, n, k),
                grad_out,
                [0, 1],
                rhs.data.as_ref(),
                [rhs.strides[1], rhs.strides[0]],
                E::ONE,
                grad_lhs,
                [0, lhs.strides[0]],
            )?;

            // grad_rhs += lhs^T * grad_out
            sgemm(
                self.blas.as_ref(),
                (k, m, n),
                lhs.data.as_ref(),
                [lhs.strides[0], 0],
                grad_out,
                [0, 1],
                E::ONE,
                grad_rhs,
                rhs.strides,
            )?;
        }
        Ok(())
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
        let mut storage = unsafe { self.dev.alloc_async::<E>(shape.num_elements()) }?;

        unsafe {
            sgemm(
                self.blas.as_ref(),
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

        Ok(self.build_tensor(shape, shape.strides(), storage))
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
        unsafe {
            // grad_lhs += grad_out * rhs^T
            sgemm(
                self.blas.as_ref(),
                (m, n, k),
                grad_out,
                strides,
                rhs.data.as_ref(),
                [rhs.strides[1], rhs.strides[0]],
                E::ONE,
                grad_lhs,
                lhs.strides,
            )?;

            // grad_rhs += lhs^T * grad_out
            sgemm(
                self.blas.as_ref(),
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
        let mut storage = unsafe { self.dev.alloc_async::<E>(shape.num_elements()) }?;
        unsafe {
            sgemm_batch(
                self.blas.as_ref(),
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
        Ok(self.build_tensor(shape, shape.strides(), storage))
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
        unsafe {
            // grad_lhs += grad_out * rhs^T
            sgemm_batch(
                self.blas.as_ref(),
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
        for i in 0..batch.size() {
            // NOTE: these have to be sequential since grad_rhs is broadcasted and cublas doesn't support
            // 0 strides with atomicAdd
            unsafe {
                // grad_rhs += lhs^T * grad_out
                sgemm(
                    self.blas.as_ref(),
                    (k, m, n),
                    &lhs.data.try_slice(i * lhs.strides[0]..).unwrap(),
                    [lhs.strides[2], lhs.strides[1]],
                    &grad_out.try_slice(i * strides[0]..).unwrap(),
                    [strides[1], strides[2]],
                    E::ONE,
                    grad_rhs,
                    rhs.strides,
                )?;
            }
        }
        Ok(())
    }
}

impl<E: Dtype> super::MatMatBatch3Kernel<E> for Cuda
where
    CudaBlas: Gemm<E>,
{
    fn forward<const B: usize, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(Const<B>, M, K), E, Self>,
        rhs: &Tensor<(Const<B>, K, N), E, Self>,
    ) -> Result<Tensor<(Const<B>, M, N), E, Self>, Self::Err> {
        assert_ne!(lhs.strides[0], 0);
        assert_ne!(rhs.strides[0], 0);
        let (batch, m, _) = lhs.shape;
        let (_, k, n) = rhs.shape;
        let shape = (batch, m, n);
        let strides = shape.strides();
        let mut storage = unsafe { self.dev.alloc_async::<E>(shape.num_elements()) }?;
        unsafe {
            sgemm_batch(
                self.blas.as_ref(),
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
        Ok(self.build_tensor(shape, shape.strides(), storage))
    }
    fn backward<const B: usize, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(Const<B>, M, K), E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<(Const<B>, K, N), E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let (batch, m, _) = lhs.shape;
        let (_, k, n) = rhs.shape;
        let strides = (batch, m, n).strides();
        unsafe {
            // grad_lhs += grad_out * rhs^T
            sgemm_batch(
                self.blas.as_ref(),
                (batch, m, n, k),
                grad_out,
                strides,
                rhs.data.as_ref(),
                [rhs.strides[0], rhs.strides[2], rhs.strides[1]],
                E::ONE,
                grad_lhs,
                lhs.strides,
            )?;

            // grad_rhs += lhs^T * grad_out
            sgemm_batch(
                self.blas.as_ref(),
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
        Ok(())
    }
}

impl<E: Dtype> super::MatMatBatch4Kernel<E> for Cuda
where
    CudaBlas: Gemm<E>,
{
    fn forward<const B: usize, const S: usize, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(Const<B>, Const<S>, M, K), E, Self>,
        rhs: &Tensor<(Const<B>, Const<S>, K, N), E, Self>,
    ) -> Result<Tensor<(Const<B>, Const<S>, M, N), E, Self>, Self::Err> {
        assert_ne!(lhs.strides[0], 0);
        assert_ne!(rhs.strides[0], 0);
        assert_ne!(lhs.strides[1], 0);
        assert_ne!(rhs.strides[1], 0);
        let (batch, seq, m, _) = lhs.shape;
        let (_, _, k, n) = rhs.shape;
        let shape = (batch, seq, m, n);
        let strides = shape.strides();
        let mut storage = unsafe { self.dev.alloc_async::<E>(shape.num_elements()) }?;

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
                    Default::default(),
                    &mut storage.try_slice_mut(b * strides[0]..).unwrap(),
                    [strides[1], strides[2], strides[3]],
                )?;
            }
        }
        Ok(self.build_tensor(shape, shape.strides(), storage))
    }

    fn backward<const B: usize, const S: usize, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(Const<B>, Const<S>, M, K), E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<(Const<B>, Const<S>, K, N), E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let (batch, seq, m, _) = lhs.shape;
        let (_, _, k, n) = rhs.shape;
        let strides = (batch, seq, m, n).strides();
        // TODO use streams
        for i in 0..batch.size() {
            unsafe {
                // gl += go * rhs^T
                sgemm_batch(
                    self.blas.as_ref(),
                    (seq, m, n, k),
                    &grad_out.try_slice(i * strides[0]..).unwrap(),
                    [strides[1], strides[2], strides[3]],
                    &rhs.data.try_slice(i * rhs.strides[0]..).unwrap(),
                    [rhs.strides[1], rhs.strides[3], rhs.strides[2]],
                    E::ONE,
                    &mut grad_lhs.try_slice_mut(i * lhs.strides[0]..).unwrap(),
                    [lhs.strides[1], lhs.strides[2], lhs.strides[3]],
                )?;

                // gr += lhs^T * go
                sgemm_batch(
                    self.blas.as_ref(),
                    (seq, k, m, n),
                    &lhs.data.try_slice(i * lhs.strides[0]..).unwrap(),
                    [lhs.strides[1], lhs.strides[3], lhs.strides[2]],
                    &grad_out.try_slice(i * strides[0]..).unwrap(),
                    [strides[1], strides[2], strides[3]],
                    E::ONE,
                    &mut grad_rhs.try_slice_mut(i * rhs.strides[0]..).unwrap(),
                    [rhs.strides[1], rhs.strides[2], rhs.strides[3]],
                )?;
            }
        }
        Ok(())
    }
}
