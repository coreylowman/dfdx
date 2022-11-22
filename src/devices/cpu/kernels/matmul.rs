use crate::arrays::*;
use crate::devices::cpu::{Cpu, View, ViewMut};
use crate::devices::{binary_ops, device::*};

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, cblas_sgemv as sgemv, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

pub(super) fn matmul<M: Dim, K: Dim, N: Dim>(
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
            layout, a_tr, b_tr, m, n, k, 1.0, a.ptr, lda, b.ptr, ldb, 1.0, c.ptr, ldc,
        )
    }
}

impl<M: Dim, const K: usize, const N: usize>
    BinaryKernel<binary_ops::MatMul, (M, C<K>), Rank2<K, N>, (M, C<N>), f32> for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<(M, C<K>), f32>,
        rhs: &Self::Storage<Rank2<K, N>, f32>,
    ) -> Result<Self::Storage<(M, C<N>), f32>, Self::Err> {
        let mut out: Self::Storage<(M, C<N>), f32> = self.try_zeros_like((lhs.shape.0, C))?;
        matmul(lhs.view(), rhs.view(), out.view_mut());
        Ok(out)
    }

    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<(M, C<K>), f32>,
        grad_lhs: &mut Self::Storage<(M, C<K>), f32>,
        rhs: &Self::Storage<Rank2<K, N>, f32>,
        grad_rhs: &mut Self::Storage<Rank2<K, N>, f32>,
        grad_out: &Self::Storage<(M, C<N>), f32>,
    ) {
        let grad_out = grad_out.view();
        matmul(grad_out, rhs.view().tr(), grad_lhs.view_mut());
        matmul(lhs.view().tr(), grad_out, grad_rhs.view_mut());
    }
}

impl<const B: usize, const M: usize, const K: usize, const N: usize>
    BinaryKernel<binary_ops::MatMul, Rank3<B, M, K>, Rank3<B, K, N>, Rank3<B, M, N>, f32> for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank3<B, M, K>, f32>,
        rhs: &Self::Storage<Rank3<B, K, N>, f32>,
    ) -> Result<Self::Storage<Rank3<B, M, N>, f32>, Self::Err> {
        let mut out: Self::Storage<Rank3<B, M, N>, f32> = self.try_zeros()?;

        let a = lhs.view();
        let b = rhs.view();
        let c = out.view_mut();

        for batch in 0..B {
            matmul(a.idx(batch), b.idx(batch), c.idx(batch));
        }

        Ok(out)
    }

    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank3<B, M, K>, f32>,
        grad_lhs: &mut Self::Storage<Rank3<B, M, K>, f32>,
        rhs: &Self::Storage<Rank3<B, K, N>, f32>,
        grad_rhs: &mut Self::Storage<Rank3<B, K, N>, f32>,
        grad_out: &Self::Storage<Rank3<B, M, N>, f32>,
    ) {
        let lhs = lhs.view();
        let grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view();
        let grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..B {
            matmul(grad_out.idx(b), rhs.idx(b).tr(), grad_lhs.idx(b));
            matmul(lhs.idx(b).tr(), grad_out.idx(b), grad_rhs.idx(b));
        }
    }
}

impl<Batch: Dim, const M: usize, const K: usize, const N: usize>
    BinaryKernel<binary_ops::MatMul, (Batch, C<M>, C<K>), Rank2<K, N>, (Batch, C<M>, C<N>), f32>
    for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<(Batch, C<M>, C<K>), f32>,
        rhs: &Self::Storage<Rank2<K, N>, f32>,
    ) -> Result<Self::Storage<(Batch, C<M>, C<N>), f32>, Self::Err> {
        let batch_size = lhs.shape().0.size();

        let mut out: Self::Storage<(Batch, C<M>, C<N>), f32> =
            self.try_zeros_like((lhs.shape().0, C, C))?;

        let a = lhs.view();
        let b = rhs.view();
        let c = out.view_mut();
        for batch in 0..batch_size {
            matmul(a.idx(batch), b, c.idx(batch));
        }

        Ok(out)
    }

    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<(Batch, C<M>, C<K>), f32>,
        grad_lhs: &mut Self::Storage<(Batch, C<M>, C<K>), f32>,
        rhs: &Self::Storage<Rank2<K, N>, f32>,
        grad_rhs: &mut Self::Storage<Rank2<K, N>, f32>,
        grad_out: &Self::Storage<(Batch, C<M>, C<N>), f32>,
    ) {
        let batch_size = lhs.shape().0.size();
        let lhs = lhs.view();
        let grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view().tr();
        let grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..batch_size {
            matmul(grad_out.idx(b), rhs, grad_lhs.idx(b));
            matmul(lhs.idx(b).tr(), grad_out.idx(b), grad_rhs);
        }
    }
}

impl<const B: usize, const S: usize, const M: usize, const K: usize, const N: usize>
    BinaryKernel<binary_ops::MatMul, Rank4<B, S, M, K>, Rank4<B, S, K, N>, Rank4<B, S, M, N>, f32>
    for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank4<B, S, M, K>, f32>,
        rhs: &Self::Storage<Rank4<B, S, K, N>, f32>,
    ) -> Result<Self::Storage<Rank4<B, S, M, N>, f32>, Self::Err> {
        let mut out: Self::Storage<Rank4<B, S, M, N>, f32> = self.try_zeros()?;
        let lhs = lhs.view();
        let rhs = rhs.view();
        let out_view = out.view_mut();
        for b in 0..B {
            for s in 0..S {
                matmul(lhs.idx(b).idx(s), rhs.idx(b).idx(s), out_view.idx(b).idx(s));
            }
        }
        Ok(out)
    }

    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank4<B, S, M, K>, f32>,
        grad_lhs: &mut Self::Storage<Rank4<B, S, M, K>, f32>,
        rhs: &Self::Storage<Rank4<B, S, K, N>, f32>,
        grad_rhs: &mut Self::Storage<Rank4<B, S, K, N>, f32>,
        grad_out: &Self::Storage<Rank4<B, S, M, N>, f32>,
    ) {
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
                );
                matmul(
                    lhs.idx(b).idx(s).tr(),
                    grad_out.idx(b).idx(s),
                    grad_rhs.idx(b).idx(s),
                );
            }
        }
    }
}

impl<const K: usize, const N: usize>
    BinaryKernel<binary_ops::MatMul, Rank1<K>, Rank2<K, N>, Rank1<N>, f32> for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank1<K>, f32>,
        rhs: &Self::Storage<Rank2<K, N>, f32>,
    ) -> Result<Self::Storage<Rank1<N>, f32>, Self::Err> {
        let mut out: Self::Storage<Rank1<N>, f32> = self.try_zeros()?;
        matmul(lhs.view().br0(), rhs.view(), out.view_mut().br0());
        Ok(out)
    }

    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank1<K>, f32>,
        grad_lhs: &mut Self::Storage<Rank1<K>, f32>,
        rhs: &Self::Storage<Rank2<K, N>, f32>,
        grad_rhs: &mut Self::Storage<Rank2<K, N>, f32>,
        grad_out: &Self::Storage<Rank1<N>, f32>,
    ) {
        let grad_out = grad_out.view().br0();
        matmul(grad_out, rhs.view().tr(), grad_lhs.view_mut().br0());
        matmul(lhs.view().br0().tr(), grad_out, grad_rhs.view_mut());
    }
}

impl<const M: usize, const N: usize>
    BinaryKernel<binary_ops::MatMul, Rank1<M>, Rank1<N>, Rank2<M, N>, f32> for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank1<M>, f32>,
        rhs: &Self::Storage<Rank1<N>, f32>,
    ) -> Result<Self::Storage<Rank2<M, N>, f32>, Self::Err> {
        let mut out: Self::Storage<Rank2<M, N>, f32> = self.try_zeros()?;
        matmul(lhs.view().br1(), rhs.view().br0(), out.view_mut());
        Ok(out)
    }
    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank1<M>, f32>,
        grad_lhs: &mut Self::Storage<Rank1<M>, f32>,
        rhs: &Self::Storage<Rank1<N>, f32>,
        grad_rhs: &mut Self::Storage<Rank1<N>, f32>,
        grad_out: &Self::Storage<Rank2<M, N>, f32>,
    ) {
        let grad_out = grad_out.view();
        matmul(grad_out, rhs.view().br0().tr(), grad_lhs.view_mut().br1());
        matmul(lhs.view().br1().tr(), grad_out, grad_rhs.view_mut().br0());
    }
}
