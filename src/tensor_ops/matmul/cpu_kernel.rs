use crate::shapes::*;
use crate::tensor::cpu::{Cpu, CpuError, StridedArray, View, ViewMut};

#[cfg(not(feature = "cblas"))]
use matrixmultiply::{dgemm, sgemm};

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_dgemm as dgemm, cblas_sgemm as sgemm, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

pub(crate) trait MatMulImpl<F> {
    fn matmul<M: Dim, K: Dim, N: Dim>(
        a: View<(M, K), F>,
        b: View<(K, N), F>,
        c: &mut ViewMut<(M, N), F>,
    );
}

impl MatMulImpl<f32> for Cpu {
    #[inline]
    fn matmul<M: Dim, K: Dim, N: Dim>(
        a: View<(M, K), f32>,
        b: View<(K, N), f32>,
        c: &mut ViewMut<(M, N), f32>,
    ) {
        let [m, k] = a.shape.concrete();
        let n = b.shape.1.size();

        let ap = a.ptr();
        let bp = b.ptr();
        let cp = c.ptr_mut();

        #[cfg(not(feature = "cblas"))]
        unsafe {
            let [ar, ac] = a.strides.map(|x| x as isize);
            let [br, bc] = b.strides.map(|x| x as isize);
            let [cr, cc] = c.strides.map(|x| x as isize);
            sgemm(m, k, n, 1.0, ap, ar, ac, bp, br, bc, 1.0, cp, cr, cc);
        }

        #[cfg(feature = "cblas")]
        unsafe {
            let (lda, a_tr) = super::matrix_strides((m, k), a.strides);
            let (ldb, b_tr) = super::matrix_strides((k, n), b.strides);
            let (ldc, c_tr) = super::matrix_strides((m, n), c.strides);
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
            )
        }
    }
}

impl MatMulImpl<f64> for Cpu {
    #[inline]
    fn matmul<M: Dim, K: Dim, N: Dim>(
        a: View<(M, K), f64>,
        b: View<(K, N), f64>,
        c: &mut ViewMut<(M, N), f64>,
    ) {
        let [m, k] = a.shape.concrete();
        let n = b.shape.1.size();

        let ap = a.ptr();
        let bp = b.ptr();
        let cp = c.ptr_mut();

        #[cfg(not(feature = "cblas"))]
        unsafe {
            let [ar, ac] = a.strides.map(|x| x as isize);
            let [br, bc] = b.strides.map(|x| x as isize);
            let [cr, cc] = c.strides.map(|x| x as isize);
            dgemm(m, k, n, 1.0, ap, ar, ac, bp, br, bc, 1.0, cp, cr, cc);
        }

        #[cfg(feature = "cblas")]
        unsafe {
            let (lda, a_tr) = super::matrix_strides((m, k), a.strides);
            let (ldb, b_tr) = super::matrix_strides((k, n), b.strides);
            let (ldc, c_tr) = super::matrix_strides((m, n), c.strides);
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
            )
        }
    }
}

impl<F: Dtype> super::VecVecKernel<F> for Cpu
where
    Self: MatMulImpl<F>,
{
    fn forward<M: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(M,), F>,
        rhs: &Self::Storage<(N,), F>,
    ) -> Result<Self::Storage<(M, N), F>, Self::Err> {
        let mut out = StridedArray::new((lhs.shape().0, rhs.shape().0))?;
        Self::matmul(lhs.view().br1(), rhs.view().br0(), &mut out.view_mut());
        Ok(out)
    }
    fn backward<M: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(M,), F>,
        grad_lhs: &mut Self::Storage<(M,), F>,
        rhs: &Self::Storage<(N,), F>,
        grad_rhs: &mut Self::Storage<(N,), F>,
        grad_out: &Self::Storage<(M, N), F>,
    ) -> Result<(), Self::Err> {
        let grad_out = grad_out.view();
        let lhs = lhs.view().br1().tr();
        let rhs = rhs.view().br0().tr();
        Self::matmul(grad_out, rhs, &mut grad_lhs.view_mut().br1());
        Self::matmul(lhs, grad_out, &mut grad_rhs.view_mut().br0());
        Ok(())
    }
}

impl<F: Dtype> super::VecMatKernel<F> for Cpu
where
    Self: MatMulImpl<F>,
{
    fn forward<const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<K>,), F>,
        rhs: &Self::Storage<(Const<K>, N), F>,
    ) -> Result<Self::Storage<(N,), F>, Self::Err> {
        let mut out = StridedArray::new((rhs.shape.1,))?;
        Self::matmul(lhs.view().br0(), rhs.view(), &mut out.view_mut().br0());
        Ok(out)
    }
    fn backward<const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<K>,), F>,
        grad_lhs: &mut Self::Storage<(Const<K>,), F>,
        rhs: &Self::Storage<(Const<K>, N), F>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), F>,
        grad_out: &Self::Storage<(N,), F>,
    ) -> Result<(), Self::Err> {
        let grad_out = grad_out.view().br0();
        Self::matmul(grad_out, rhs.view().tr(), &mut grad_lhs.view_mut().br0());
        Self::matmul(lhs.view().br0().tr(), grad_out, &mut grad_rhs.view_mut());
        Ok(())
    }
}

impl<F: Dtype> super::MatMatKernel<F> for Cpu
where
    Self: MatMulImpl<F>,
{
    fn forward<M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(M, Const<K>), F>,
        rhs: &Self::Storage<(Const<K>, N), F>,
    ) -> Result<Self::Storage<(M, N), F>, Self::Err> {
        let mut out = StridedArray::new((lhs.shape.0, rhs.shape.1))?;
        Self::matmul(lhs.view(), rhs.view(), &mut out.view_mut());
        Ok(out)
    }
    fn backward<M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(M, Const<K>), F>,
        grad_lhs: &mut Self::Storage<(M, Const<K>), F>,
        rhs: &Self::Storage<(Const<K>, N), F>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), F>,
        grad_out: &Self::Storage<(M, N), F>,
    ) -> Result<(), Self::Err> {
        let grad_out = grad_out.view();
        Self::matmul(grad_out, rhs.view().tr(), &mut grad_lhs.view_mut());
        Self::matmul(lhs.view().tr(), grad_out, &mut grad_rhs.view_mut());
        Ok(())
    }
}

impl<F: Dtype> super::MatMatBrKernel<F> for Cpu
where
    Self: MatMulImpl<F>,
{
    fn forward<B: Dim, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(B, M, Const<K>), F>,
        rhs: &Self::Storage<(Const<K>, N), F>,
    ) -> Result<Self::Storage<(B, M, N), F>, Self::Err> {
        let (batch, seq, _) = *lhs.shape();
        let (_, n) = *rhs.shape();
        let mut out = StridedArray::new((batch, seq, n))?;
        let a = lhs.view();
        let b = rhs.view();
        let mut c = out.view_mut();
        for batch in 0..batch.size() {
            Self::matmul(a.idx(batch), b, &mut c.idx_mut(batch));
        }
        Ok(out)
    }
    fn backward<B: Dim, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(B, M, Const<K>), F>,
        grad_lhs: &mut Self::Storage<(B, M, Const<K>), F>,
        rhs: &Self::Storage<(Const<K>, N), F>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), F>,
        grad_out: &Self::Storage<(B, M, N), F>,
    ) -> Result<(), Self::Err> {
        let batch_size = lhs.shape().0.size();
        let lhs = lhs.view();
        let mut grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view().tr();
        let mut grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..batch_size {
            let go = grad_out.idx(b);
            Self::matmul(go, rhs, &mut grad_lhs.idx_mut(b));
            Self::matmul(lhs.idx(b).tr(), go, &mut grad_rhs);
        }
        Ok(())
    }
}

impl<F: Dtype> super::MatMatBatch3Kernel<F> for Cpu
where
    Self: MatMulImpl<F>,
{
    fn forward<const B: usize, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, M, K), F>,
        rhs: &Self::Storage<(Const<B>, K, N), F>,
    ) -> Result<Self::Storage<(Const<B>, M, N), F>, Self::Err> {
        let m: M = lhs.shape().1;
        let n: N = rhs.shape().2;

        let k: K = lhs.shape().2;
        let k2: K = rhs.shape().1;
        if k != k2 {
            return Err(CpuError::WrongNumElements);
        }
        let mut out = StridedArray::new((Const, m, n))?;
        let a = lhs.view();
        let b = rhs.view();
        let mut c = out.view_mut();
        for batch in 0..B {
            Self::matmul(a.idx(batch), b.idx(batch), &mut c.idx_mut(batch));
        }
        Ok(out)
    }
    fn backward<const B: usize, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, M, K), F>,
        grad_lhs: &mut Self::Storage<(Const<B>, M, K), F>,
        rhs: &Self::Storage<(Const<B>, K, N), F>,
        grad_rhs: &mut Self::Storage<(Const<B>, K, N), F>,
        grad_out: &Self::Storage<(Const<B>, M, N), F>,
    ) -> Result<(), Self::Err> {
        let lhs = lhs.view();
        let mut grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view();
        let mut grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..B {
            let go = grad_out.idx(b);
            Self::matmul(go, rhs.idx(b).tr(), &mut grad_lhs.idx_mut(b));
            Self::matmul(lhs.idx(b).tr(), go, &mut grad_rhs.idx_mut(b));
        }
        Ok(())
    }
}

impl<F: Dtype> super::MatMatBatch4Kernel<F> for Cpu
where
    Self: MatMulImpl<F>,
{
    fn forward<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, Const<S>, M, Const<K>), F>,
        rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), F>,
    ) -> Result<Self::Storage<(Const<B>, Const<S>, M, N), F>, Self::Err> {
        let m: M = lhs.shape.2;
        let n: N = rhs.shape.3;
        let mut out = StridedArray::new((Const, Const, m, n))?;
        let lhs = lhs.view();
        let rhs = rhs.view();
        let mut out_view = out.view_mut();
        for b in 0..B {
            let l_b = lhs.idx(b);
            let r_b = rhs.idx(b);
            let mut o_b = out_view.idx_mut(b);
            for s in 0..S {
                Self::matmul(l_b.idx(s), r_b.idx(s), &mut o_b.idx_mut(s));
            }
        }
        Ok(out)
    }
    fn backward<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, Const<S>, M, Const<K>), F>,
        grad_lhs: &mut Self::Storage<(Const<B>, Const<S>, M, Const<K>), F>,
        rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), F>,
        grad_rhs: &mut Self::Storage<(Const<B>, Const<S>, Const<K>, N), F>,
        grad_out: &Self::Storage<(Const<B>, Const<S>, M, N), F>,
    ) -> Result<(), Self::Err> {
        let lhs = lhs.view();
        let mut grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view();
        let mut grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..B {
            let l_b = lhs.idx(b);
            let mut gl_b = grad_lhs.idx_mut(b);
            let r_b = rhs.idx(b);
            let mut gr_b = grad_rhs.idx_mut(b);
            let go_b = grad_out.idx(b);
            for s in 0..S {
                Self::matmul(go_b.idx(s), r_b.idx(s).tr(), &mut gl_b.idx_mut(s));
                Self::matmul(l_b.idx(s).tr(), go_b.idx(s), &mut gr_b.idx_mut(s));
            }
        }
        Ok(())
    }
}
