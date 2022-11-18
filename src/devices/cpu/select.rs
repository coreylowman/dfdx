use super::device::{Cpu, StridedArray};
use super::iterate::LendingIterator;
use crate::arrays::{Axis, Dim, Dtype, ReduceShape, Shape};
use crate::devices::device::*;

// (M, ) -> ()
impl<E: Dtype + std::ops::AddAssign, D1: Dim>
    UnaryKernel<unary_ops::Select<(), Axis<0>, (), Self>, (D1,), (), E> for Cpu
{
    fn unary_fwd(
        &self,
        op: unary_ops::Select<(), Axis<0>, (), Self>,
        inp: &Self::Storage<(D1,), E>,
    ) -> Result<Self::Storage<(), E>, Self::Err> {
        let mut out: StridedArray<(), E> = self.try_zeros()?;
        let i = op.indices[[]];
        out[[]] = inp[[i]];
        Ok(out)
    }

    fn unary_bwd(
        &self,
        op: unary_ops::Select<(), Axis<0>, (), Self>,
        _inp: &Self::Storage<(D1,), E>,
        grad_inp: &mut Self::Storage<(D1,), E>,
        grad_out: &Self::Storage<(), E>,
    ) {
        let i = op.indices[[]];
        grad_inp[[i]] += grad_out[[]];
    }
}

// (M, ) -> (Z, )
impl<E: Dtype + std::ops::AddAssign, Src: Dim, Dst: Dim>
    UnaryKernel<unary_ops::Select<(Dst,), Axis<0>, (Dst,), Self>, (Src,), (Dst,), E> for Cpu
{
    fn unary_fwd(
        &self,
        op: unary_ops::Select<(Dst,), Axis<0>, (Dst,), Self>,
        inp: &Self::Storage<(Src,), E>,
    ) -> Result<Self::Storage<(Dst,), E>, Self::Err> {
        let mut out: StridedArray<(Dst,), E> = self.try_zeros_like(op.dst)?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((o, i)) = out_iter.next() {
            let reidx = [op.indices[i]];
            *o = inp[reidx];
        }
        Ok(out)
    }
    fn unary_bwd(
        &self,
        op: unary_ops::Select<(Dst,), Axis<0>, (Dst,), Self>,
        inp: &Self::Storage<(Src,), E>,
        grad_inp: &mut Self::Storage<(Src,), E>,
        grad_out: &Self::Storage<(Dst,), E>,
    ) {
    }
}

// (M, N) -> (N, )
impl<E: Dtype + std::ops::AddAssign, D1: Dim, D2: Dim>
    UnaryKernel<unary_ops::Select<(D2,), Axis<0>, (), Self>, (D1, D2), (D2,), E> for Cpu
{
    fn unary_fwd(
        &self,
        op: unary_ops::Select<(D2,), Axis<0>, (), Self>,
        inp: &Self::Storage<(D1, D2), E>,
    ) -> Result<Self::Storage<(D2,), E>, Self::Err> {
        let mut out: StridedArray<(D2,), E> = self.try_zeros_like(op.dst)?;
        let mut out_iter = out.iter_mut_with_index();
        let idx = op.indices[[]];
        while let Some((o, i)) = out_iter.next() {
            let reidx = [idx, i[0]];
            *o = inp[reidx];
        }
        Ok(out)
    }
    fn unary_bwd(
        &self,
        op: unary_ops::Select<(D2,), Axis<0>, (), Self>,
        inp: &Self::Storage<(D1, D2), E>,
        grad_inp: &mut Self::Storage<(D1, D2), E>,
        grad_out: &Self::Storage<(D2,), E>,
    ) {
    }
}

// (M, N) -> (M, )
impl<E: Dtype + std::ops::AddAssign, D1: Dim, D2: Dim>
    UnaryKernel<unary_ops::Select<(D1,), Axis<1>, (), Self>, (D1, D2), (D1,), E> for Cpu
{
    fn unary_fwd(
        &self,
        op: unary_ops::Select<(D1,), Axis<1>, (), Self>,
        inp: &Self::Storage<(D1, D2), E>,
    ) -> Result<Self::Storage<(D1,), E>, Self::Err> {
        let mut out: StridedArray<(D1,), E> = self.try_zeros_like(op.dst)?;
        let mut out_iter = out.iter_mut_with_index();
        let idx = op.indices[[]];
        while let Some((o, i)) = out_iter.next() {
            let reidx = [i[0], idx];
            *o = inp[reidx];
        }
        Ok(out)
    }
    fn unary_bwd(
        &self,
        op: unary_ops::Select<(D1,), Axis<1>, (), Self>,
        inp: &Self::Storage<(D1, D2), E>,
        grad_inp: &mut Self::Storage<(D1, D2), E>,
        grad_out: &Self::Storage<(D1,), E>,
    ) {
    }
}

// (M, N) -> (Z, N)
impl<E: Dtype + std::ops::AddAssign, Seq: Dim, Src1: Dim, Src2: Dim>
    UnaryKernel<unary_ops::Select<(Seq, Src2), Axis<0>, (Seq,), Self>, (Src1, Src2), (Seq, Src2), E>
    for Cpu
{
    fn unary_fwd(
        &self,
        op: unary_ops::Select<(Seq, Src2), Axis<0>, (Seq,), Self>,
        inp: &Self::Storage<(Src1, Src2), E>,
    ) -> Result<Self::Storage<(Seq, Src2), E>, Self::Err> {
        let mut out: StridedArray<(Seq, Src2), E> = self.try_zeros_like(op.dst)?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((o, i)) = out_iter.next() {
            let reidx = [op.indices[[i[0]]], i[1]];
            *o = inp[reidx];
        }
        Ok(out)
    }
    fn unary_bwd(
        &self,
        op: unary_ops::Select<(Seq, Src2), Axis<0>, (Seq,), Self>,
        inp: &Self::Storage<(Src1, Src2), E>,
        grad_inp: &mut Self::Storage<(Src1, Src2), E>,
        grad_out: &Self::Storage<(Seq, Src2), E>,
    ) {
    }
}

// (M, N) -> (B, Z, N)
impl<E: Dtype + std::ops::AddAssign, Batch: Dim, Seq: Dim, Src1: Dim, Src2: Dim>
    UnaryKernel<
        unary_ops::Select<(Batch, Seq, Src2), Axis<0>, (Batch, Seq), Self>,
        (Src1, Src2),
        (Batch, Seq, Src2),
        E,
    > for Cpu
{
    fn unary_fwd(
        &self,
        op: unary_ops::Select<(Batch, Seq, Src2), Axis<0>, (Batch, Seq), Self>,
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
        op: unary_ops::Select<(Batch, Seq, Src2), Axis<0>, (Batch, Seq), Self>,
        _inp: &Self::Storage<(Src1, Src2), E>,
        grad_inp: &mut Self::Storage<(Src1, Src2), E>,
        grad_out: &Self::Storage<(Batch, Seq, Src2), E>,
    ) {
        let mut out_iter = grad_out.iter_with_index();
        while let Some((o, [b, s, i])) = out_iter.next() {
            let idx = op.indices[[b, s]];
            grad_inp[[idx, i]] += *o;
        }
    }
}
