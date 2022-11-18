use super::device::Cpu;
use crate::arrays::*;
use crate::devices::device::*;

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
                matrixmultiply::sgemm(M, K, N, 1.0, a, ar, ac, b, bc, br, 1.0, c, cr, cc);
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
                matrixmultiply::sgemm(M, K, N, 1.0, a, ac, ar, b, br, bc, 1.0, c, cr, cc);
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
