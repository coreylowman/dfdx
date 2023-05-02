#![allow(clippy::needless_return)]

use crate::shapes::*;
use crate::tensor::{Cpu, Tensor, ZerosTensor};

use std::sync::Arc;

#[cfg(feature = "cpu-mkl-matmul")]
use cblas_sys::{
    cblas_dgemm as dgemm, cblas_sgemm as sgemm, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

#[cfg(all(
    any(feature = "cpu-seq-matmul", feature = "cpu-par-matmul"),
    not(feature = "cpu-mkl-matmul")
))]
use matrixmultiply::{dgemm, sgemm};

#[allow(unused)]
fn naive_gemm<F: num_traits::Float + std::ops::AddAssign, M: Dim, K: Dim, N: Dim>(
    (m, k, n): (M, K, N),
    ap: *const F,
    a_strides: [usize; 2],
    bp: *const F,
    b_strides: [usize; 2],
    cp: *mut F,
    c_strides: [usize; 2],
) {
    for i_m in 0..m.size() {
        for i_k in 0..k.size() {
            for i_n in 0..n.size() {
                unsafe {
                    let a = *ap.add(a_strides[0] * i_m + a_strides[1] * i_k);
                    let b = *bp.add(b_strides[0] * i_k + b_strides[1] * i_n);
                    let c = cp.add(c_strides[0] * i_m + c_strides[1] * i_n);
                    *c += a * b;
                }
            }
        }
    }
}

pub(crate) trait MatMulImpl<E> {
    fn matmul<M: Dim, K: Dim, N: Dim>(
        dims: (M, K, N),
        ap: *const E,
        a_strides: [usize; 2],
        bp: *const E,
        b_strides: [usize; 2],
        cp: *mut E,
        c_strides: [usize; 2],
    );
}

#[cfg(feature = "f16")]
impl MatMulImpl<half::f16> for Cpu {
    #[inline]
    fn matmul<M: Dim, K: Dim, N: Dim>(
        (m, k, n): (M, K, N),
        ap: *const half::f16,
        a_strides: [usize; 2],
        bp: *const half::f16,
        b_strides: [usize; 2],
        cp: *mut half::f16,
        c_strides: [usize; 2],
    ) {
        const CHUNK_SIZE: usize = 8;
        let mut a_chunk: [f32; CHUNK_SIZE] = Default::default();
        let mut b_chunk: [[f32; CHUNK_SIZE]; CHUNK_SIZE] = Default::default();
        let mut c_chunk: [f32; CHUNK_SIZE] = Default::default();

        for i_m in 0..m.size() {
            let a_m = unsafe { ap.add(a_strides[0] * i_m) };
            let c_m = unsafe { cp.add(c_strides[0] * i_m) };
            for i_n_base in (0..n.size()).step_by(CHUNK_SIZE) {
                let n_chunk_size = CHUNK_SIZE.min(n.size() - i_n_base);

                {
                    // load c into chunk
                    for i_chunk_n in 0..n_chunk_size {
                        let c_mn = unsafe { *c_m.add(c_strides[1] * (i_n_base + i_chunk_n)) };
                        c_chunk[i_chunk_n] = c_mn.into();
                    }
                    for i_chunk_n in n_chunk_size..CHUNK_SIZE {
                        c_chunk[i_chunk_n] = 0.0;
                    }
                }

                for i_k_base in (0..k.size()).step_by(CHUNK_SIZE) {
                    let k_chunk_size = CHUNK_SIZE.min(k.size() - i_k_base);

                    {
                        // load a & b into chunk
                        for i_chunk_k in 0..k_chunk_size {
                            let i_k = i_k_base + i_chunk_k;

                            // load a
                            let a_mk = unsafe { *a_m.add(a_strides[1] * i_k) };
                            a_chunk[i_chunk_k] = a_mk.into();

                            // load b
                            for i_chunk_n in 0..n_chunk_size {
                                let i_n = i_n_base + i_chunk_n;
                                let b_kn =
                                    unsafe { *bp.add(b_strides[0] * i_k + b_strides[1] * i_n) };
                                b_chunk[i_chunk_k][i_chunk_n] = b_kn.into();
                            }
                            for i_chunk_n in n_chunk_size..CHUNK_SIZE {
                                b_chunk[i_chunk_k][i_chunk_n] = 0.0;
                            }
                        }
                        for i_chunk_k in k_chunk_size..CHUNK_SIZE {
                            a_chunk[i_chunk_k] = 0.0;
                            for i_chunk_n in n_chunk_size..CHUNK_SIZE {
                                b_chunk[i_chunk_k][i_chunk_n] = 0.0;
                            }
                        }
                    }

                    // do the computation
                    for i_n_chunk in 0..CHUNK_SIZE {
                        for i_k_chunk in 0..CHUNK_SIZE {
                            c_chunk[i_n_chunk] +=
                                a_chunk[i_k_chunk] * b_chunk[i_k_chunk][i_n_chunk];
                        }
                    }
                }

                // store c chunk in memory
                for i_chunk in 0..n_chunk_size {
                    let r = half::f16::from_f32(c_chunk[i_chunk]);
                    unsafe { *c_m.add(c_strides[1] * (i_n_base + i_chunk)) = r };
                }
            }
        }
    }
}

impl MatMulImpl<f32> for Cpu {
    #[inline]
    fn matmul<M: Dim, K: Dim, N: Dim>(
        (m, k, n): (M, K, N),
        ap: *const f32,
        a_strides: [usize; 2],
        bp: *const f32,
        b_strides: [usize; 2],
        cp: *mut f32,
        c_strides: [usize; 2],
    ) {
        let (m, k, n) = (m.size(), k.size(), n.size());
        #[cfg(feature = "cpu-mkl-matmul")]
        unsafe {
            let (lda, a_tr) = super::matrix_strides((m, k), a_strides);
            let (ldb, b_tr) = super::matrix_strides((k, n), b_strides);
            let (ldc, c_tr) = super::matrix_strides((m, n), c_strides);
            let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);
            let layout = if c_tr { ColMajor } else { RowMajor };
            let (a_tr, b_tr) = if c_tr {
                (if a_tr { NoTr } else { Tr }, if b_tr { NoTr } else { Tr })
            } else {
                (if a_tr { Tr } else { NoTr }, if b_tr { Tr } else { NoTr })
            };
            sgemm(
                layout, a_tr, b_tr, m, n, k, 1.0, ap, lda as i32, bp, ldb as i32, 1.0, cp,
                ldc as i32,
            );
        }

        #[cfg(all(
            any(feature = "cpu-seq-matmul", feature = "cpu-par-matmul"),
            not(feature = "cpu-mkl-matmul")
        ))]
        unsafe {
            let [ar, ac] = a_strides.map(|x| x as isize);
            let [br, bc] = b_strides.map(|x| x as isize);
            let [cr, cc] = c_strides.map(|x| x as isize);
            sgemm(m, k, n, 1.0, ap, ar, ac, bp, br, bc, 1.0, cp, cr, cc);
        }

        #[cfg(not(any(
            feature = "cpu-seq-matmul",
            feature = "cpu-par-matmul",
            feature = "cpu-mkl-matmul"
        )))]
        naive_gemm((m, k, n), ap, a_strides, bp, b_strides, cp, c_strides);
    }
}

impl MatMulImpl<f64> for Cpu {
    #[inline]
    fn matmul<M: Dim, K: Dim, N: Dim>(
        (m, k, n): (M, K, N),
        ap: *const f64,
        a_strides: [usize; 2],
        bp: *const f64,
        b_strides: [usize; 2],
        cp: *mut f64,
        c_strides: [usize; 2],
    ) {
        let (m, k, n) = (m.size(), k.size(), n.size());

        #[cfg(feature = "cpu-mkl-matmul")]
        unsafe {
            let (lda, a_tr) = super::matrix_strides((m, k), a_strides);
            let (ldb, b_tr) = super::matrix_strides((k, n), b_strides);
            let (ldc, c_tr) = super::matrix_strides((m, n), c_strides);
            let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);
            let layout = if c_tr { ColMajor } else { RowMajor };
            let (a_tr, b_tr) = if c_tr {
                (if a_tr { NoTr } else { Tr }, if b_tr { NoTr } else { Tr })
            } else {
                (if a_tr { Tr } else { NoTr }, if b_tr { Tr } else { NoTr })
            };
            dgemm(
                layout, a_tr, b_tr, m, n, k, 1.0, ap, lda as i32, bp, ldb as i32, 1.0, cp,
                ldc as i32,
            );
            return;
        }

        #[cfg(all(
            any(feature = "cpu-seq-matmul", feature = "cpu-par-matmul"),
            not(feature = "cpu-mkl-matmul")
        ))]
        unsafe {
            let [ar, ac] = a_strides.map(|x| x as isize);
            let [br, bc] = b_strides.map(|x| x as isize);
            let [cr, cc] = c_strides.map(|x| x as isize);
            dgemm(m, k, n, 1.0, ap, ar, ac, bp, br, bc, 1.0, cp, cr, cc);
            return;
        }

        #[cfg(not(any(
            feature = "cpu-seq-matmul",
            feature = "cpu-par-matmul",
            feature = "cpu-mkl-matmul"
        )))]
        naive_gemm((m, k, n), ap, a_strides, bp, b_strides, cp, c_strides);
    }
}

impl<E: Dtype> super::MatMatKernel<E> for Cpu
where
    Self: MatMulImpl<E>,
{
    fn forward<M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(M, K), E, Self>,
        rhs: &Tensor<(K, N), E, Self>,
    ) -> Result<Tensor<(M, N), E, Self>, Self::Err> {
        let (m, k) = lhs.shape;
        let n = rhs.shape.1;
        let mut out = self.try_zeros_like(&(m, n))?;
        Self::matmul(
            (m, k, n),
            lhs.data.as_ptr(),
            lhs.strides,
            rhs.data.as_ptr(),
            rhs.strides,
            Arc::get_mut(&mut out.data).unwrap().as_mut_ptr(),
            out.strides,
        );
        Ok(out)
    }
    fn backward<M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(M, K), E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<(K, N), E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let (m, k) = lhs.shape;
        let n = rhs.shape.1;
        let strides = (m, n).strides();
        Self::matmul(
            (m, n, k),
            grad_out.as_ptr(),
            strides,
            rhs.data.as_ptr(),
            [rhs.strides[1], rhs.strides[0]],
            grad_lhs.as_mut_ptr(),
            lhs.strides,
        );
        Self::matmul(
            (k, m, n),
            lhs.data.as_ptr(),
            [lhs.strides[1], lhs.strides[0]],
            grad_out.as_ptr(),
            strides,
            grad_rhs.as_mut_ptr(),
            rhs.strides,
        );
        Ok(())
    }
}

impl<E: Dtype> super::MatMatBrKernel<E> for Cpu
where
    Self: MatMulImpl<E>,
{
    fn forward<B: Dim, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(B, M, K), E, Self>,
        rhs: &Tensor<(K, N), E, Self>,
    ) -> Result<Tensor<(B, M, N), E, Self>, Self::Err> {
        let (batch, m, k) = lhs.shape;
        let n = rhs.shape.1;
        let mut out = self.try_zeros_like(&(batch, m, n))?;
        let cp = Arc::get_mut(&mut out.data).unwrap();
        for i in 0..batch.size() {
            Self::matmul(
                (m, k, n),
                lhs.data[i * lhs.strides[0]..].as_ptr(),
                [lhs.strides[1], lhs.strides[2]],
                rhs.data.as_ptr(),
                rhs.strides,
                cp[i * out.strides[0]..].as_mut_ptr(),
                [out.strides[1], out.strides[2]],
            );
        }
        Ok(out)
    }
    fn backward<B: Dim, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(B, M, K), E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<(K, N), E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let (batch, m, k) = lhs.shape;
        let n = rhs.shape.1;
        let strides = (batch, m, n).strides();
        for i in 0..batch.size() {
            Self::matmul(
                (m, n, k),
                grad_out[i * strides[0]..].as_ptr(),
                [strides[1], strides[2]],
                rhs.data.as_ptr(),
                [rhs.strides[1], rhs.strides[0]],
                grad_lhs[i * lhs.strides[0]..].as_mut_ptr(),
                [lhs.strides[1], lhs.strides[2]],
            );
            Self::matmul(
                (k, m, n),
                lhs.data[i * lhs.strides[0]..].as_ptr(),
                [lhs.strides[2], lhs.strides[1]],
                grad_out[i * strides[0]..].as_ptr(),
                [strides[1], strides[2]],
                grad_rhs.as_mut_ptr(),
                rhs.strides,
            );
        }
        Ok(())
    }
}

impl<E: Dtype> super::MatMatBatch3Kernel<E> for Cpu
where
    Self: MatMulImpl<E>,
{
    fn forward<B: Dim, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(B, M, K), E, Self>,
        rhs: &Tensor<(B, K, N), E, Self>,
    ) -> Result<Tensor<(B, M, N), E, Self>, Self::Err> {
        let (b, m, k) = lhs.shape;
        let n = rhs.shape.2;
        let mut out = self.try_zeros_like(&(b, m, n))?;
        let ap = lhs.data.as_ref();
        let bp = rhs.data.as_ref();
        let cp = Arc::get_mut(&mut out.data).unwrap();
        for i in 0..b.size() {
            Self::matmul(
                (m, k, n),
                ap[i * lhs.strides[0]..].as_ptr(),
                [lhs.strides[1], lhs.strides[2]],
                bp[i * rhs.strides[0]..].as_ptr(),
                [rhs.strides[1], rhs.strides[2]],
                cp[i * out.strides[0]..].as_mut_ptr(),
                [out.strides[1], out.strides[2]],
            )
        }
        Ok(out)
    }
    fn backward<B: Dim, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(B, M, K), E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<(B, K, N), E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let (b, m, k) = lhs.shape;
        let n = rhs.shape.2;
        let strides = (b, m, n).strides();
        for i in 0..b.size() {
            Self::matmul(
                (m, n, k),
                grad_out[i * strides[0]..].as_ptr(),
                [strides[1], strides[2]],
                rhs.data[i * rhs.strides[0]..].as_ptr(),
                [rhs.strides[2], rhs.strides[1]],
                grad_lhs[i * lhs.strides[0]..].as_mut_ptr(),
                [lhs.strides[1], lhs.strides[2]],
            );
            Self::matmul(
                (k, m, n),
                lhs.data[i * lhs.strides[0]..].as_ptr(),
                [lhs.strides[2], lhs.strides[1]],
                grad_out[i * strides[0]..].as_ptr(),
                [strides[1], strides[2]],
                grad_rhs[i * rhs.strides[0]..].as_mut_ptr(),
                [rhs.strides[1], rhs.strides[2]],
            );
        }
        Ok(())
    }
}

impl<E: Dtype> super::MatMatBatch4Kernel<E> for Cpu
where
    Self: MatMulImpl<E>,
{
    fn forward<B: Dim, S: Dim, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(B, S, M, K), E, Self>,
        rhs: &Tensor<(B, S, K, N), E, Self>,
    ) -> Result<Tensor<(B, S, M, N), E, Self>, Self::Err> {
        let (b, s, m, k) = lhs.shape;
        let n = rhs.shape.3;
        let mut out = self.try_zeros_like(&(b, s, m, n))?;
        let cp = Arc::get_mut(&mut out.data).unwrap();
        for i in 0..b.size() {
            for j in 0..s.size() {
                Self::matmul(
                    (m, k, n),
                    lhs.data[i * lhs.strides[0] + j * lhs.strides[1]..].as_ptr(),
                    [lhs.strides[2], lhs.strides[3]],
                    rhs.data[i * rhs.strides[0] + j * rhs.strides[1]..].as_ptr(),
                    [rhs.strides[2], rhs.strides[3]],
                    cp[i * out.strides[0] + j * out.strides[1]..].as_mut_ptr(),
                    [out.strides[2], out.strides[3]],
                );
            }
        }
        Ok(out)
    }
    fn backward<B: Dim, S: Dim, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Tensor<(B, S, M, K), E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<(B, S, K, N), E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let (b, s, m, k) = lhs.shape;
        let n = rhs.shape.3;
        let strides = (b, s, m, n).strides();
        for i in 0..b.size() {
            for j in 0..s.size() {
                Self::matmul(
                    (m, n, k),
                    grad_out[i * strides[0] + j * strides[1]..].as_ptr(),
                    [strides[2], strides[3]],
                    rhs.data[i * rhs.strides[0] + j * rhs.strides[1]..].as_ptr(),
                    [rhs.strides[3], rhs.strides[2]],
                    grad_lhs[i * lhs.strides[0] + j * lhs.strides[1]..].as_mut_ptr(),
                    [lhs.strides[2], lhs.strides[3]],
                );
                Self::matmul(
                    (k, m, n),
                    lhs.data[i * lhs.strides[0] + j * lhs.strides[1]..].as_ptr(),
                    [lhs.strides[3], lhs.strides[2]],
                    grad_out[i * strides[0] + j * strides[1]..].as_ptr(),
                    [strides[2], strides[3]],
                    grad_rhs[i * rhs.strides[0] + j * rhs.strides[1]..].as_mut_ptr(),
                    [rhs.strides[2], rhs.strides[3]],
                );
            }
        }
        Ok(())
    }
}
