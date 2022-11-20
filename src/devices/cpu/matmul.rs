use super::device::Cpu;
use crate::arrays::*;
use crate::devices::{binary_ops, device::*};

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, cblas_sgemv as sgemv, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

impl<const M: usize, const K: usize, const N: usize>
    BinaryKernel<binary_ops::MatMul, Rank2<M, K>, Rank2<K, N>, Rank2<M, N>, f32> for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank2<M, K>, f32>,
        rhs: &Self::Storage<Rank2<K, N>, f32>,
    ) -> Result<Self::Storage<Rank2<M, N>, f32>, Self::Err> {
        let mut out: Self::Storage<Rank2<M, N>, f32> = self.try_zeros()?;

        let (a, a_strides) = lhs.strided_ptr();
        let (b, b_strides) = rhs.strided_ptr();
        let (c, c_strides) = out.strided_ptr_mut();

        #[cfg(not(feature = "cblas"))]
        unsafe {
            let [ar, ac] = a_strides.map(|x| x as isize);
            let [br, bc] = b_strides.map(|x| x as isize);
            let [cr, cc] = c_strides.map(|x| x as isize);
            matrixmultiply::sgemm(M, K, N, 1.0, a, ar, ac, b, br, bc, 1.0, c, cr, cc);
        }

        #[cfg(feature = "cblas")]
        unsafe {
            let (m, n, k) = (M as libc::c_int, N as libc::c_int, K as libc::c_int);
            let [ar, _] = a_strides.map(|x| x as libc::c_int);
            let [br, _] = b_strides.map(|x| x as libc::c_int);
            let [cr, _] = c_strides.map(|x| x as libc::c_int);
            sgemm(RowMajor, NoTr, NoTr, m, n, k, 1.0, a, ar, b, br, 1.0, c, cr)
        }

        Ok(out)
    }

    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank2<M, K>, f32>,
        grad_lhs: &mut Self::Storage<Rank2<M, K>, f32>,
        rhs: &Self::Storage<Rank2<K, N>, f32>,
        grad_rhs: &mut Self::Storage<Rank2<K, N>, f32>,
        grad_out: &Self::Storage<Rank2<M, N>, f32>,
    ) {
        {
            // grad_lhs += grad_out * rhs^T
            // (M, K) += (M, N) * (N, K)
            let (a, a_strides) = grad_out.strided_ptr();
            let (b, b_strides) = rhs.strided_ptr();
            let (c, c_strides) = grad_lhs.strided_ptr_mut();

            #[cfg(not(feature = "cblas"))]
            unsafe {
                let [ar, ac] = a_strides.map(|x| x as isize);
                let [br, bc] = b_strides.map(|x| x as isize);
                let [cr, cc] = c_strides.map(|x| x as isize);
                matrixmultiply::sgemm(M, N, K, 1.0, a, ar, ac, b, bc, br, 1.0, c, cr, cc);
            }

            #[cfg(feature = "cblas")]
            unsafe {
                let [ar, _] = a_strides.map(|x| x as libc::c_int);
                let [br, _] = b_strides.map(|x| x as libc::c_int);
                let [cr, _] = c_strides.map(|x| x as libc::c_int);
                let (m, n, k) = (M as libc::c_int, N as libc::c_int, K as libc::c_int);
                sgemm(RowMajor, NoTr, Tr, m, k, n, 1.0, a, ar, b, br, 1.0, c, cr)
            }
        }

        {
            // grad_rhs += lhs^T * grad_out
            // (K, N) += (K, M) * (M, N)
            let (a, a_strides) = lhs.strided_ptr();
            let (b, b_strides) = grad_out.strided_ptr();
            let (c, c_strides) = grad_rhs.strided_ptr_mut();

            #[cfg(not(feature = "cblas"))]
            unsafe {
                let [ar, ac] = a_strides.map(|x| x as isize);
                let [br, bc] = b_strides.map(|x| x as isize);
                let [cr, cc] = c_strides.map(|x| x as isize);
                matrixmultiply::sgemm(K, M, N, 1.0, a, ac, ar, b, br, bc, 1.0, c, cr, cc);
            }

            #[cfg(feature = "cblas")]
            unsafe {
                let [ar, _] = a_strides.map(|x| x as libc::c_int);
                let [br, _] = b_strides.map(|x| x as libc::c_int);
                let [cr, _] = c_strides.map(|x| x as libc::c_int);
                let (m, n, k) = (M as libc::c_int, N as libc::c_int, K as libc::c_int);
                sgemm(RowMajor, Tr, NoTr, k, n, m, 1.0, a, ar, b, br, 1.0, c, cr)
            }
        }
    }
}

impl<Batch: Dim, const M: usize, const K: usize, const N: usize>
    BinaryKernel<
        binary_ops::MatMul,
        (Batch, C<M>, C<K>),
        (Batch, C<K>, C<N>),
        (Batch, C<M>, C<N>),
        f32,
    > for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<(Batch, C<M>, C<K>), f32>,
        rhs: &Self::Storage<(Batch, C<K>, C<N>), f32>,
    ) -> Result<Self::Storage<(Batch, C<M>, C<N>), f32>, Self::Err> {
        todo!();
    }
    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<(Batch, C<M>, C<K>), f32>,
        grad_lhs: &mut Self::Storage<(Batch, C<M>, C<K>), f32>,
        rhs: &Self::Storage<(Batch, C<K>, C<N>), f32>,
        grad_rhs: &mut Self::Storage<(Batch, C<K>, C<N>), f32>,
        grad_out: &Self::Storage<(Batch, C<M>, C<N>), f32>,
    ) {
    }
}

impl<Batch: Dim, const M: usize, const K: usize, const N: usize>
    BinaryKernel<binary_ops::MatMul, (Batch, C<M>, C<K>), (C<K>, C<N>), (Batch, C<M>, C<N>), f32>
    for Cpu
{
    fn binary_fwd(
        &self,
        op: binary_ops::MatMul,
        lhs: &Self::Storage<(Batch, C<M>, C<K>), f32>,
        rhs: &Self::Storage<(C<K>, C<N>), f32>,
    ) -> Result<Self::Storage<(Batch, C<M>, C<N>), f32>, Self::Err> {
        todo!();
    }

    fn binary_bwd(
        &self,
        op: binary_ops::MatMul,
        lhs: &Self::Storage<(Batch, C<M>, C<K>), f32>,
        grad_lhs: &mut Self::Storage<(Batch, C<M>, C<K>), f32>,
        rhs: &Self::Storage<(C<K>, C<N>), f32>,
        grad_rhs: &mut Self::Storage<(C<K>, C<N>), f32>,
        grad_out: &Self::Storage<(Batch, C<M>, C<N>), f32>,
    ) {
        todo!();
    }
}

impl<Batch: Dim, Seq: Dim, const M: usize, const K: usize, const N: usize>
    BinaryKernel<
        binary_ops::MatMul,
        (Batch, Seq, C<M>, C<K>),
        (Batch, Seq, C<K>, C<N>),
        (Batch, Seq, C<M>, C<N>),
        f32,
    > for Cpu
{
    fn binary_fwd(
        &self,
        op: binary_ops::MatMul,
        lhs: &Self::Storage<(Batch, Seq, C<M>, C<K>), f32>,
        rhs: &Self::Storage<(Batch, Seq, C<K>, C<N>), f32>,
    ) -> Result<Self::Storage<(Batch, Seq, C<M>, C<N>), f32>, Self::Err> {
        todo!();
    }
    fn binary_bwd(
        &self,
        op: binary_ops::MatMul,
        lhs: &Self::Storage<(Batch, Seq, C<M>, C<K>), f32>,
        grad_lhs: &mut Self::Storage<(Batch, Seq, C<M>, C<K>), f32>,
        rhs: &Self::Storage<(Batch, Seq, C<K>, C<N>), f32>,
        grad_rhs: &mut Self::Storage<(Batch, Seq, C<K>, C<N>), f32>,
        grad_out: &Self::Storage<(Batch, Seq, C<M>, C<N>), f32>,
    ) {
        todo!();
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
        todo!();
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
        todo!();
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
    }
}
