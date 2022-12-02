use crate::arrays::*;
use crate::tensor::cpu::{Cpu, StridedArray, View, ViewMut};

use super::{
    MatMatBatch3Kernel, MatMatBatch4Kernel, MatMatBrKernel, MatMatKernel, VecMatKernel,
    VecVecKernel,
};

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

#[inline]
pub(crate) fn matmul<M: Dim, K: Dim, N: Dim>(
    a: View<(M, K), f32>,
    b: View<(K, N), f32>,
    c: ViewMut<(M, N), f32>,
) {
    let [m, k] = a.shape.concrete();
    let n = b.shape.1.size();

    #[cfg(not(feature = "cblas"))]
    unsafe {
        let [ar, ac] = a.strides.map(|x| x as isize);
        let [br, bc] = b.strides.map(|x| x as isize);
        let [cr, cc] = c.strides.map(|x| x as isize);
        matrixmultiply::sgemm(
            m, k, n, 1.0, a.ptr, ar, ac, b.ptr, br, bc, 1.0, c.ptr, cr, cc,
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
            layout, a_tr, b_tr, m, n, k, a.ptr, lda, b.ptr, ldb, 1.0, c.ptr, ldc,
        )
    }
}

impl VecVecKernel<f32> for Cpu {
    fn forward<M: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(M,), f32>,
        rhs: &Self::Storage<(N,), f32>,
    ) -> Result<Self::Storage<(M, N), f32>, Self::Err> {
        let mut out = StridedArray::new((lhs.shape().0, rhs.shape().0))?;
        matmul(lhs.view().br1(), rhs.view().br0(), out.view_mut());
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
        matmul(grad_out, rhs.view().br0().tr(), grad_lhs.view_mut().br1());
        matmul(lhs.view().br1().tr(), grad_out, grad_rhs.view_mut().br0());
        Ok(())
    }
}

impl VecMatKernel<f32> for Cpu {
    fn forward<const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<K>,), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
    ) -> Result<Self::Storage<(N,), f32>, Self::Err> {
        let mut out = StridedArray::new((rhs.shape.1,))?;
        matmul(lhs.view().br0(), rhs.view(), out.view_mut().br0());
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
        matmul(grad_out, rhs.view().tr(), grad_lhs.view_mut().br0());
        matmul(lhs.view().br0().tr(), grad_out, grad_rhs.view_mut());
        Ok(())
    }
}

impl MatMatKernel<f32> for Cpu {
    fn forward<M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
    ) -> Result<Self::Storage<(M, N), f32>, Self::Err> {
        let mut out = StridedArray::new((lhs.shape.0, rhs.shape.1))?;
        matmul(lhs.view(), rhs.view(), out.view_mut());
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
        matmul(grad_out, rhs.view().tr(), grad_lhs.view_mut());
        matmul(lhs.view().tr(), grad_out, grad_rhs.view_mut());
        Ok(())
    }
}

impl MatMatBrKernel<f32> for Cpu {
    fn forward<B: Dim, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(B, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
    ) -> Result<Self::Storage<(B, M, N), f32>, Self::Err> {
        let &(batch, seq, _) = lhs.shape();
        let &(_, n) = rhs.shape();
        let mut out = StridedArray::new((batch, seq, n))?;
        let a = lhs.view();
        let b = rhs.view();
        let c = out.view_mut();
        for batch in 0..batch.size() {
            matmul(a.idx(batch), b, c.idx(batch));
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
            let go = grad_out.idx(b);
            matmul(go, rhs, grad_lhs.idx(b));
            matmul(lhs.idx(b).tr(), go, grad_rhs);
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
        let mut out = StridedArray::new((Const, *m, *n))?;
        let a = lhs.view();
        let b = rhs.view();
        let c = out.view_mut();
        for batch in 0..B {
            matmul(a.idx(batch), b.idx(batch), c.idx(batch));
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
            let go = grad_out.idx(b);
            matmul(go, rhs.idx(b).tr(), grad_lhs.idx(b));
            matmul(lhs.idx(b).tr(), go, grad_rhs.idx(b));
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
        let mut out = StridedArray::new((Const, Const, *m, *n))?;
        let lhs = lhs.view();
        let rhs = rhs.view();
        let out_view = out.view_mut();
        for b in 0..B {
            let l_b = lhs.idx(b);
            let r_b = rhs.idx(b);
            let o_b = out_view.idx(b);
            for s in 0..S {
                matmul(l_b.idx(s), r_b.idx(s), o_b.idx(s));
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
            let l_b = lhs.idx(b);
            let gl_b = grad_lhs.idx(b);
            let r_b = rhs.idx(b);
            let gr_b = grad_rhs.idx(b);
            let go_b = grad_out.idx(b);
            for s in 0..S {
                matmul(go_b.idx(s), r_b.idx(s).tr(), gl_b.idx(s));
                matmul(l_b.idx(s).tr(), go_b.idx(s), gr_b.idx(s));
            }
        }
        Ok(())
    }
}
