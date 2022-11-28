use crate::arrays::*;
use crate::devices::{
    cpu::{Cpu, View, ViewMut},
    ZerosLike,
};

use super::{
    MatMatBatch3Kernel, MatMatBatch4Kernel, MatMatBrKernel, MatMatKernel, VecMatKernel,
    VecVecKernel,
};

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, cblas_sgemv as sgemv, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

pub(crate) fn matmul<M: Dim, K: Dim, N: Dim>(
    a: View<(M, K), f32>,
    b: View<(K, N), f32>,
    c: ViewMut<(M, N), f32>,
    beta: f32,
) {
    let [m, k] = a.shape.concrete();
    let n = b.shape.1.size();

    #[cfg(not(feature = "cblas"))]
    unsafe {
        let [ar, ac] = a.strides.map(|x| x as isize);
        let [br, bc] = b.strides.map(|x| x as isize);
        let [cr, cc] = c.strides.map(|x| x as isize);
        matrixmultiply::sgemm(
            m, k, n, 1.0, a.ptr, ar, ac, b.ptr, br, bc, beta, c.ptr, cr, cc,
        );
    }

    #[cfg(feature = "cblas")]
    unsafe {
        let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);
        let [ar, ac] = a.strides.map(|x| x as libc::c_int);
        let [br, bc] = b.strides.map(|x| x as libc::c_int);
        let [cr, cc] = c.strides.map(|x| x as libc::c_int);
        let (layout, a_tr, b_tr, lda, ldb, ldc) = if cr < cc {
            let (lda, a_tr) = if ar < ac { (m, NoTr) } else { (k, Tr) };
            let (ldb, b_tr) = if br < bc { (k, NoTr) } else { (n, Tr) };
            (ColMajor, a_tr, b_tr, lda, ldb, m)
        } else {
            let (lda, a_tr) = if ar < ac { (m, Tr) } else { (k, NoTr) };
            let (ldb, b_tr) = if br < bc { (k, Tr) } else { (n, NoTr) };
            (RowMajor, a_tr, b_tr, lda, ldb, n)
        };
        sgemm(
            layout, a_tr, b_tr, m, n, k, 1.0, a.ptr, lda, b.ptr, ldb, beta, c.ptr, ldc,
        )
    }
}

impl VecVecKernel<f32> for Cpu {
    fn forward<M: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(M,), f32>,
        rhs: &Self::Storage<(N,), f32>,
    ) -> Result<Self::Storage<(M, N), f32>, Self::Err> {
        let mut out: Self::Storage<(M, N), f32> =
            self.try_zeros_like((lhs.shape().0, rhs.shape().0))?;
        matmul(lhs.view().br1(), rhs.view().br0(), out.view_mut(), 1.0);
        Ok(out)
    }
    fn backward<M: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(M,), f32>,
        grad_lhs: &mut Self::Storage<(M,), f32>,
        rhs: &Self::Storage<(N,), f32>,
        grad_rhs: &mut Self::Storage<(N,), f32>,
        grad_out: &Self::Storage<(M, N), f32>,
    ) -> Result<(), Self::Err> {
        let grad_out = grad_out.view();
        matmul(
            grad_out,
            rhs.view().br0().tr(),
            grad_lhs.view_mut().br1(),
            1.0,
        );
        matmul(
            lhs.view().br1().tr(),
            grad_out,
            grad_rhs.view_mut().br0(),
            1.0,
        );
        Ok(())
    }
}

impl VecMatKernel<f32> for Cpu {
    fn forward<const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<K>,), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
    ) -> Result<Self::Storage<(N,), f32>, Self::Err> {
        let (_, n) = rhs.shape();
        let mut out: Self::Storage<(N,), f32> = self.try_zeros_like((*n,))?;
        matmul(lhs.view().br0(), rhs.view(), out.view_mut().br0(), 1.0);
        Ok(out)
    }
    fn backward<const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<K>,), f32>,
        grad_lhs: &mut Self::Storage<(Const<K>,), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), f32>,
        grad_out: &Self::Storage<(N,), f32>,
    ) -> Result<(), Self::Err> {
        let grad_out = grad_out.view().br0();
        matmul(grad_out, rhs.view().tr(), grad_lhs.view_mut().br0(), 1.0);
        matmul(lhs.view().br0().tr(), grad_out, grad_rhs.view_mut(), 1.0);
        Ok(())
    }
}

impl MatMatKernel<f32> for Cpu {
    fn forward<M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
    ) -> Result<Self::Storage<(M, N), f32>, Self::Err> {
        let (m, _) = lhs.shape();
        let (_, n) = rhs.shape();
        let mut out: Self::Storage<(M, N), f32> = self.try_zeros_like((*m, *n))?;
        matmul(lhs.view(), rhs.view(), out.view_mut(), 1.0);
        Ok(out)
    }
    fn backward<M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(M, Const<K>), f32>,
        grad_lhs: &mut Self::Storage<(M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), f32>,
        grad_out: &Self::Storage<(M, N), f32>,
    ) -> Result<(), Self::Err> {
        let grad_out = grad_out.view();
        matmul(grad_out, rhs.view().tr(), grad_lhs.view_mut(), 1.0);
        matmul(lhs.view().tr(), grad_out, grad_rhs.view_mut(), 1.0);
        Ok(())
    }
}

impl MatMatBrKernel<f32> for Cpu {
    fn forward<B: Dim, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(B, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
    ) -> Result<Self::Storage<(B, M, N), f32>, Self::Err> {
        let (batch, seq, _) = lhs.shape();
        let (_, n) = rhs.shape();
        let mut out: Self::Storage<(B, M, N), f32> = self.try_zeros_like((*batch, *seq, *n))?;
        let a = lhs.view();
        let b = rhs.view();
        let c = out.view_mut();
        for batch in 0..batch.size() {
            matmul(a.idx(batch), b, c.idx(batch), 1.0);
        }

        Ok(out)
    }
    fn backward<B: Dim, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(B, M, Const<K>), f32>,
        grad_lhs: &mut Self::Storage<(B, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), f32>,
        grad_out: &Self::Storage<(B, M, N), f32>,
    ) -> Result<(), Self::Err> {
        let batch_size = lhs.shape().0.size();
        let lhs = lhs.view();
        let grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view().tr();
        let grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..batch_size {
            matmul(grad_out.idx(b), rhs, grad_lhs.idx(b), 1.0);
            matmul(lhs.idx(b).tr(), grad_out.idx(b), grad_rhs, 1.0);
        }
        Ok(())
    }
}

impl MatMatBatch3Kernel<f32> for Cpu {
    fn forward<const B: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<K>, N), f32>,
    ) -> Result<Self::Storage<(Const<B>, M, N), f32>, Self::Err> {
        let (_, m, _) = lhs.shape();
        let (_, _, n) = rhs.shape();
        let mut out: Self::Storage<(Const<B>, M, N), f32> = self.try_zeros_like((Const, *m, *n))?;
        let a = lhs.view();
        let b = rhs.view();
        let c = out.view_mut();
        for batch in 0..B {
            matmul(a.idx(batch), b.idx(batch), c.idx(batch), 1.0);
        }
        Ok(out)
    }
    fn backward<const B: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, M, Const<K>), f32>,
        grad_lhs: &mut Self::Storage<(Const<B>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<B>, Const<K>, N), f32>,
        grad_out: &Self::Storage<(Const<B>, M, N), f32>,
    ) -> Result<(), Self::Err> {
        let lhs = lhs.view();
        let grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view();
        let grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..B {
            matmul(grad_out.idx(b), rhs.idx(b).tr(), grad_lhs.idx(b), 1.0);
            matmul(lhs.idx(b).tr(), grad_out.idx(b), grad_rhs.idx(b), 1.0);
        }
        Ok(())
    }
}

impl MatMatBatch4Kernel<f32> for Cpu {
    fn forward<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
    ) -> Result<Self::Storage<(Const<B>, Const<S>, M, N), f32>, Self::Err> {
        let (_, _, m, _) = lhs.shape();
        let (_, _, _, n) = rhs.shape();
        let mut out: Self::Storage<(Const<B>, Const<S>, M, N), f32> =
            self.try_zeros_like((Const, Const, *m, *n))?;
        let lhs = lhs.view();
        let rhs = rhs.view();
        let out_view = out.view_mut();
        for b in 0..B {
            for s in 0..S {
                matmul(
                    lhs.idx(b).idx(s),
                    rhs.idx(b).idx(s),
                    out_view.idx(b).idx(s),
                    1.0,
                );
            }
        }
        Ok(out)
    }
    fn backward<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
        grad_lhs: &mut Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
        grad_rhs: &mut Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
        grad_out: &Self::Storage<(Const<B>, Const<S>, M, N), f32>,
    ) -> Result<(), Self::Err> {
        let lhs = lhs.view();
        let grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view();
        let grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..B {
            for s in 0..S {
                matmul(
                    grad_out.idx(b).idx(s),
                    rhs.idx(b).idx(s).tr(),
                    grad_lhs.idx(b).idx(s),
                    1.0,
                );
                matmul(
                    lhs.idx(b).idx(s).tr(),
                    grad_out.idx(b).idx(s),
                    grad_rhs.idx(b).idx(s),
                    1.0,
                );
            }
        }
        Ok(())
    }
}
