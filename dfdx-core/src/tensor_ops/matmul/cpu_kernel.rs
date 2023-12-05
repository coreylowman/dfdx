#![allow(clippy::needless_return)]

use crate::shapes::*;
use crate::tensor::{Cpu, Error, Tensor, ZerosTensor};

use std::sync::Arc;

#[allow(unused)]
#[allow(clippy::too_many_arguments)]
fn naive_gemm<F: num_traits::Float + std::ops::AddAssign, M: Dim, K: Dim, N: Dim>(
    (m, k, n): (M, K, N),
    accum: bool,
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
                    if accum {
                        *c += a * b;
                    } else {
                        *c = a * b;
                    }
                }
            }
        }
    }
}

pub(crate) trait MatMulImpl<E> {
    #[allow(clippy::too_many_arguments)]
    fn matmul<M: Dim, K: Dim, N: Dim>(
        dims: (M, K, N),
        accum: bool,
        ap: *const E,
        a_strides: [usize; 2],
        bp: *const E,
        b_strides: [usize; 2],
        cp: *mut E,
        c_strides: [usize; 2],
    );
}

#[cfg(feature = "f16")]
impl MatMulImpl<crate::dtypes::AMP<half::f16>> for Cpu {
    #[inline]
    fn matmul<M: Dim, K: Dim, N: Dim>(
        (m, k, n): (M, K, N),
        accum: bool,
        ap: *const crate::dtypes::AMP<half::f16>,
        astr: [usize; 2],
        bp: *const crate::dtypes::AMP<half::f16>,
        bstr: [usize; 2],
        cp: *mut crate::dtypes::AMP<half::f16>,
        cstr: [usize; 2],
    ) {
        #[cfg(not(feature = "cpu"))]
        naive_gemm((m, k, n), accum, ap, astr, bp, bstr, cp, cstr);

        #[cfg(feature = "cpu")]
        unsafe {
            gemm::gemm(
                m.size(),
                n.size(),
                k.size(),
                cp as *mut gemm::f16,
                cstr[1] as isize,
                cstr[0] as isize,
                accum,
                ap as *const gemm::f16,
                astr[1] as isize,
                astr[0] as isize,
                bp as *const gemm::f16,
                bstr[1] as isize,
                bstr[0] as isize,
                if accum {
                    gemm::f16::ONE
                } else {
                    gemm::f16::ZERO
                },
                gemm::f16::ONE,
                false,
                false,
                false,
                gemm::Parallelism::None,
            )
        }
    }
}

#[cfg(feature = "f16")]
impl MatMulImpl<half::f16> for Cpu {
    #[inline]
    fn matmul<M: Dim, K: Dim, N: Dim>(
        (m, k, n): (M, K, N),
        accum: bool,
        ap: *const half::f16,
        astr: [usize; 2],
        bp: *const half::f16,
        bstr: [usize; 2],
        cp: *mut half::f16,
        cstr: [usize; 2],
    ) {
        #[cfg(not(feature = "cpu"))]
        naive_gemm((m, k, n), accum, ap, astr, bp, bstr, cp, cstr);

        #[cfg(feature = "cpu")]
        #[allow(clippy::unnecessary_cast)]
        unsafe {
            gemm::gemm(
                m.size(),
                n.size(),
                k.size(),
                cp as *mut gemm::f16,
                cstr[1] as isize,
                cstr[0] as isize,
                accum,
                ap as *const gemm::f16,
                astr[1] as isize,
                astr[0] as isize,
                bp as *const gemm::f16,
                bstr[1] as isize,
                bstr[0] as isize,
                if accum {
                    gemm::f16::ONE
                } else {
                    gemm::f16::ZERO
                },
                gemm::f16::ONE,
                false,
                false,
                false,
                gemm::Parallelism::None,
            )
        }
    }
}

impl MatMulImpl<f32> for Cpu {
    #[inline]
    fn matmul<M: Dim, K: Dim, N: Dim>(
        (m, k, n): (M, K, N),
        accum: bool,
        ap: *const f32,
        astr: [usize; 2],
        bp: *const f32,
        bstr: [usize; 2],
        cp: *mut f32,
        cstr: [usize; 2],
    ) {
        #[cfg(not(feature = "cpu"))]
        naive_gemm((m, k, n), accum, ap, astr, bp, bstr, cp, cstr);

        #[cfg(feature = "cpu")]
        unsafe {
            gemm::gemm(
                m.size(),
                n.size(),
                k.size(),
                cp,
                cstr[1] as isize,
                cstr[0] as isize,
                accum,
                ap,
                astr[1] as isize,
                astr[0] as isize,
                bp,
                bstr[1] as isize,
                bstr[0] as isize,
                if accum { 1.0 } else { 0.0 },
                1.0,
                false,
                false,
                false,
                gemm::Parallelism::None,
            )
        }
    }
}

impl MatMulImpl<f64> for Cpu {
    #[inline]
    fn matmul<M: Dim, K: Dim, N: Dim>(
        (m, k, n): (M, K, N),
        accum: bool,
        ap: *const f64,
        astr: [usize; 2],
        bp: *const f64,
        bstr: [usize; 2],
        cp: *mut f64,
        cstr: [usize; 2],
    ) {
        #[cfg(not(feature = "cpu"))]
        naive_gemm((m, k, n), accum, ap, astr, bp, bstr, cp, cstr);

        #[cfg(feature = "cpu")]
        unsafe {
            gemm::gemm(
                m.size(),
                n.size(),
                k.size(),
                cp,
                cstr[1] as isize,
                cstr[0] as isize,
                accum,
                ap,
                astr[1] as isize,
                astr[0] as isize,
                bp,
                bstr[1] as isize,
                bstr[0] as isize,
                if accum { 1.0 } else { 0.0 },
                1.0,
                false,
                false,
                false,
                gemm::Parallelism::None,
            )
        }
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
    ) -> Result<Tensor<(M, N), E, Self>, Error> {
        let (m, k) = lhs.shape;
        let n = rhs.shape.1;
        let mut out = self.try_zeros_like(&(m, n))?;
        Self::matmul(
            (m, k, n),
            false,
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
        grad_lhs: &mut Self::Vec,
        rhs: &Tensor<(K, N), E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        let (m, k) = lhs.shape;
        let n = rhs.shape.1;
        let strides = (m, n).strides();
        Self::matmul(
            (m, n, k),
            true,
            grad_out.as_ptr(),
            strides,
            rhs.data.as_ptr(),
            [rhs.strides[1], rhs.strides[0]],
            grad_lhs.as_mut_ptr(),
            lhs.strides,
        );
        Self::matmul(
            (k, m, n),
            true,
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
    ) -> Result<Tensor<(B, M, N), E, Self>, Error> {
        let (batch, m, k) = lhs.shape;
        let n = rhs.shape.1;
        let mut out = self.try_zeros_like(&(batch, m, n))?;
        let cp = Arc::get_mut(&mut out.data).unwrap();
        for i in 0..batch.size() {
            Self::matmul(
                (m, k, n),
                false,
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
        grad_lhs: &mut Self::Vec,
        rhs: &Tensor<(K, N), E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        let (batch, m, k) = lhs.shape;
        let n = rhs.shape.1;
        let strides = (batch, m, n).strides();
        for i in 0..batch.size() {
            Self::matmul(
                (m, n, k),
                true,
                grad_out[i * strides[0]..].as_ptr(),
                [strides[1], strides[2]],
                rhs.data.as_ptr(),
                [rhs.strides[1], rhs.strides[0]],
                grad_lhs[i * lhs.strides[0]..].as_mut_ptr(),
                [lhs.strides[1], lhs.strides[2]],
            );
            Self::matmul(
                (k, m, n),
                true,
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
    ) -> Result<Tensor<(B, M, N), E, Self>, Error> {
        let (b, m, k) = lhs.shape;
        let n = rhs.shape.2;
        let mut out = self.try_zeros_like(&(b, m, n))?;
        let ap = lhs.data.as_ref();
        let bp = rhs.data.as_ref();
        let cp = Arc::get_mut(&mut out.data).unwrap();
        for i in 0..b.size() {
            Self::matmul(
                (m, k, n),
                false,
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
        grad_lhs: &mut Self::Vec,
        rhs: &Tensor<(B, K, N), E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        let (b, m, k) = lhs.shape;
        let n = rhs.shape.2;
        let strides = (b, m, n).strides();
        for i in 0..b.size() {
            Self::matmul(
                (m, n, k),
                true,
                grad_out[i * strides[0]..].as_ptr(),
                [strides[1], strides[2]],
                rhs.data[i * rhs.strides[0]..].as_ptr(),
                [rhs.strides[2], rhs.strides[1]],
                grad_lhs[i * lhs.strides[0]..].as_mut_ptr(),
                [lhs.strides[1], lhs.strides[2]],
            );
            Self::matmul(
                (k, m, n),
                true,
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
    ) -> Result<Tensor<(B, S, M, N), E, Self>, Error> {
        let (b, s, m, k) = lhs.shape;
        let n = rhs.shape.3;
        let mut out = self.try_zeros_like(&(b, s, m, n))?;
        let cp = Arc::get_mut(&mut out.data).unwrap();
        for i in 0..b.size() {
            for j in 0..s.size() {
                Self::matmul(
                    (m, k, n),
                    false,
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
        grad_lhs: &mut Self::Vec,
        rhs: &Tensor<(B, S, K, N), E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        let (b, s, m, k) = lhs.shape;
        let n = rhs.shape.3;
        let strides = (b, s, m, n).strides();
        for i in 0..b.size() {
            for j in 0..s.size() {
                Self::matmul(
                    (m, n, k),
                    true,
                    grad_out[i * strides[0] + j * strides[1]..].as_ptr(),
                    [strides[2], strides[3]],
                    rhs.data[i * rhs.strides[0] + j * rhs.strides[1]..].as_ptr(),
                    [rhs.strides[3], rhs.strides[2]],
                    grad_lhs[i * lhs.strides[0] + j * lhs.strides[1]..].as_mut_ptr(),
                    [lhs.strides[2], lhs.strides[3]],
                );
                Self::matmul(
                    (k, m, n),
                    true,
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
