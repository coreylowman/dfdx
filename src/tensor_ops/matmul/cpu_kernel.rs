use crate::shapes::*;
use crate::tensor::cpu::{Cpu, StridedArray, View, ViewMut};

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

#[inline]
pub(crate) fn matmul<M: Dim, K: Dim, N: Dim>(
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
        matrixmultiply::sgemm(m, k, n, 1.0, ap, ar, ac, bp, br, bc, 1.0, cp, cr, cc);
    }

    #[cfg(feature = "cblas")]
    unsafe {
        let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);

        let (lda, a_tr) = match a.strides {
            [1, 0] => (m as i32, true),
            [0, 1] => (k as i32, false),
            [ld, 1] => (ld as i32, false),
            [1, ld] => (ld as i32, true),
            _ => panic!("At least one of a's strides must be 1 for cblas"),
        };

        let (ldb, b_tr) = match b.strides {
            [1, 0] => (k as i32, true),
            [0, 1] => (n as i32, false),
            [ld, 1] => (ld as i32, false),
            [1, ld] => (ld as i32, true),
            _ => panic!("At least one of b's strides must be 1 for cblas"),
        };

        let (ldc, c_trans) = match c.strides {
            [1, 0] => (m as i32, true),
            [0, 1] => (n as i32, false),
            [ld, 1] => (ld as i32, false),
            [1, ld] => (ld as i32, true),
            _ => panic!("At least one of c's strides must be 1 for cblas"),
        };

        let layout = if c_trans { ColMajor } else { RowMajor };
        let (a_tr, b_tr) = if c_trans {
            (if a_tr { NoTr } else { Tr }, if b_tr { NoTr } else { Tr })
        } else {
            (if a_tr { Tr } else { NoTr }, if b_tr { Tr } else { NoTr })
        };
        sgemm(
            layout, a_tr, b_tr, m, n, k, 1.0, ap, lda, bp, ldb, 1.0, cp, ldc,
        )
    }
}

impl super::VecVecKernel<f32> for Cpu {
    fn forward<M: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(M,), f32>,
        rhs: &Self::Storage<(N,), f32>,
    ) -> Result<Self::Storage<(M, N), f32>, Self::Err> {
        let mut out = StridedArray::new((lhs.shape().0, rhs.shape().0))?;
        matmul(lhs.view().br1(), rhs.view().br0(), &mut out.view_mut());
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
        let lhs = lhs.view().br1().tr();
        let rhs = rhs.view().br0().tr();
        matmul(grad_out, rhs, &mut grad_lhs.view_mut().br1());
        matmul(lhs, grad_out, &mut grad_rhs.view_mut().br0());
        Ok(())
    }
}

impl super::VecMatKernel<f32> for Cpu {
    fn forward<const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<K>,), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
    ) -> Result<Self::Storage<(N,), f32>, Self::Err> {
        let mut out = StridedArray::new((rhs.shape.1,))?;
        matmul(lhs.view().br0(), rhs.view(), &mut out.view_mut().br0());
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
        matmul(grad_out, rhs.view().tr(), &mut grad_lhs.view_mut().br0());
        matmul(lhs.view().br0().tr(), grad_out, &mut grad_rhs.view_mut());
        Ok(())
    }
}

impl super::MatMatKernel<f32> for Cpu {
    fn forward<M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
    ) -> Result<Self::Storage<(M, N), f32>, Self::Err> {
        let mut out = StridedArray::new((lhs.shape.0, rhs.shape.1))?;
        matmul(lhs.view(), rhs.view(), &mut out.view_mut());
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
        matmul(grad_out, rhs.view().tr(), &mut grad_lhs.view_mut());
        matmul(lhs.view().tr(), grad_out, &mut grad_rhs.view_mut());
        Ok(())
    }
}

impl super::MatMatBrKernel<f32> for Cpu {
    fn forward<B: Dim, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(B, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<K>, N), f32>,
    ) -> Result<Self::Storage<(B, M, N), f32>, Self::Err> {
        let (batch, seq, _) = *lhs.shape();
        let (_, n) = *rhs.shape();
        let mut out = StridedArray::new((batch, seq, n))?;
        let a = lhs.view();
        let b = rhs.view();
        let mut c = out.view_mut();
        for batch in 0..batch.size() {
            matmul(a.idx(batch), b, &mut c.idx_mut(batch));
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
        let mut grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view().tr();
        let mut grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..batch_size {
            let go = grad_out.idx(b);
            matmul(go, rhs, &mut grad_lhs.idx_mut(b));
            matmul(lhs.idx(b).tr(), go, &mut grad_rhs);
        }
        Ok(())
    }
}

impl super::MatMatBatch3Kernel<f32> for Cpu {
    fn forward<const B: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<K>, N), f32>,
    ) -> Result<Self::Storage<(Const<B>, M, N), f32>, Self::Err> {
        let m: M = lhs.shape().1;
        let n: N = rhs.shape().2;
        let mut out = StridedArray::new((Const, m, n))?;
        let a = lhs.view();
        let b = rhs.view();
        let mut c = out.view_mut();
        for batch in 0..B {
            matmul(a.idx(batch), b.idx(batch), &mut c.idx_mut(batch));
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
        let mut grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view();
        let mut grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..B {
            let go = grad_out.idx(b);
            matmul(go, rhs.idx(b).tr(), &mut grad_lhs.idx_mut(b));
            matmul(lhs.idx(b).tr(), go, &mut grad_rhs.idx_mut(b));
        }
        Ok(())
    }
}

impl super::MatMatBatch4Kernel<f32> for Cpu {
    fn forward<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
        rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
    ) -> Result<Self::Storage<(Const<B>, Const<S>, M, N), f32>, Self::Err> {
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
                matmul(l_b.idx(s), r_b.idx(s), &mut o_b.idx_mut(s));
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
                matmul(go_b.idx(s), r_b.idx(s).tr(), &mut gl_b.idx_mut(s));
                matmul(l_b.idx(s).tr(), go_b.idx(s), &mut gr_b.idx_mut(s));
            }
        }
        Ok(())
    }
}
