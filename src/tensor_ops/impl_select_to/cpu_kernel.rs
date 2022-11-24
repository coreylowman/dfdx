#![allow(clippy::needless_range_loop)]

use crate::arrays::{Axis, Dim, Dtype, ReduceShape, ReplaceDim, Shape};
use crate::devices::{
    cpu::{Cpu, LendingIterator, StridedArray},
    ZerosLike,
};
use crate::tensor_ops::utils::UnaryKernel;

use super::SelectKernelOp;

// select reduce a single axis
impl<
        const I: isize,
        const N: usize,
        const M: usize,
        E: Dtype + std::ops::AddAssign,
        S: Shape<Concrete = [usize; N]> + ReduceShape<Axis<I>>,
    > UnaryKernel<SelectKernelOp<S::Reduced, Axis<I>, (), Self>, S, S::Reduced, E> for Cpu
where
    S::Reduced: Shape<Concrete = [usize; M]>,
{
    fn unary_fwd(
        &self,
        op: SelectKernelOp<S::Reduced, Axis<I>, (), Self>,
        inp: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S::Reduced, E>, Self::Err> {
        let mut out: StridedArray<S::Reduced, E> = self.try_zeros_like(op.dst)?;
        let mut out_iter = out.iter_mut_with_index();
        let idx = op.indices[[]];
        while let Some((o, i)) = out_iter.next() {
            let mut j = 0;
            let mut reidx: [usize; N] = [0; N];
            for n in 0..N {
                if n == I as usize {
                    reidx[n] = idx;
                } else {
                    reidx[n] = i[j];
                    j += 1;
                }
            }
            *o = inp[reidx];
        }
        Ok(out)
    }
    fn unary_bwd(
        &self,
        op: SelectKernelOp<S::Reduced, Axis<I>, (), Self>,
        _inp: &Self::Storage<S, E>,
        grad_inp: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S::Reduced, E>,
    ) -> Result<(), Self::Err> {
        let mut out_iter = grad_out.iter_with_index();
        let idx = op.indices[[]];
        while let Some((o, i)) = out_iter.next() {
            let mut j = 0;
            let mut reidx: [usize; N] = [0; N];
            for n in 0..N {
                if n == I as usize {
                    reidx[n] = idx;
                } else {
                    reidx[n] = i[j];
                    j += 1;
                }
            }
            grad_inp[reidx] += *o;
        }
        Ok(())
    }
}

// replace an axis with a new size
impl<
        const I: isize,
        const N: usize,
        New: Dim,
        E: Dtype + std::ops::AddAssign,
        S: Shape<Concrete = [usize; N]> + ReplaceDim<I, New>,
    > UnaryKernel<SelectKernelOp<S::Replaced, Axis<I>, (New,), Self>, S, S::Replaced, E> for Cpu
{
    fn unary_fwd(
        &self,
        op: SelectKernelOp<S::Replaced, Axis<I>, (New,), Self>,
        inp: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S::Replaced, E>, Self::Err> {
        let mut out: StridedArray<S::Replaced, E> = self.try_zeros_like(op.dst)?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((o, mut i)) = out_iter.next() {
            let dim_i_idx = i[I as usize];
            i[I as usize] = op.indices[[dim_i_idx]];
            *o = inp[i];
        }
        Ok(out)
    }
    fn unary_bwd(
        &self,
        op: SelectKernelOp<S::Replaced, Axis<I>, (New,), Self>,
        _inp: &Self::Storage<S, E>,
        grad_inp: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S::Replaced, E>,
    ) -> Result<(), Self::Err> {
        let mut out_iter = grad_out.iter_with_index();
        while let Some((o, mut i)) = out_iter.next() {
            let dim_i_idx = i[I as usize];
            i[I as usize] = op.indices[[dim_i_idx]];
            grad_inp[i] += *o;
        }
        Ok(())
    }
}

// batched select (M, N) -> (B, Z, N)
impl<E: Dtype + std::ops::AddAssign, Batch: Dim, Seq: Dim, Src1: Dim, Src2: Dim>
    UnaryKernel<
        SelectKernelOp<(Batch, Seq, Src2), Axis<0>, (Batch, Seq), Self>,
        (Src1, Src2),
        (Batch, Seq, Src2),
        E,
    > for Cpu
{
    fn unary_fwd(
        &self,
        op: SelectKernelOp<(Batch, Seq, Src2), Axis<0>, (Batch, Seq), Self>,
        inp: &Self::Storage<(Src1, Src2), E>,
    ) -> Result<Self::Storage<(Batch, Seq, Src2), E>, Self::Err> {
        let mut out: StridedArray<(Batch, Seq, Src2), E> = self.try_zeros_like(op.dst)?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((o, [b, s, i])) = out_iter.next() {
            let idx = op.indices[[b, s]];
            *o = inp[[idx, i]];
        }
        Ok(out)
    }

    fn unary_bwd(
        &self,
        op: SelectKernelOp<(Batch, Seq, Src2), Axis<0>, (Batch, Seq), Self>,
        _inp: &Self::Storage<(Src1, Src2), E>,
        grad_inp: &mut Self::Storage<(Src1, Src2), E>,
        grad_out: &Self::Storage<(Batch, Seq, Src2), E>,
    ) -> Result<(), Self::Err> {
        let mut out_iter = grad_out.iter_with_index();
        while let Some((o, [b, s, i])) = out_iter.next() {
            let idx = op.indices[[b, s]];
            grad_inp[[idx, i]] += *o;
        }
        Ok(())
    }
}
