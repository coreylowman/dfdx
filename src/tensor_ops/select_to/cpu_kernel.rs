#![allow(clippy::needless_range_loop)]

use crate::arrays::{Axis, Dim, Dtype, ReduceShape, ReplaceDim, Shape};
use crate::devices::Zeros;
use crate::devices::{
    cpu::{Cpu, LendingIterator, StridedArray},
    ZerosLike,
};

use super::{ReplaceAxisKernel, SelectAxisKernel, SelectBatchKernel};

// select reduce a single axis
impl<E: Dtype> SelectAxisKernel<E> for Cpu {
    fn forward<const I: isize, S: Shape + ReduceShape<Axis<I>>>(
        &self,
        inp: &Self::Storage<S, E>,
        idx: &Self::Storage<(), usize>,
    ) -> Result<Self::Storage<S::Reduced, E>, Self::Err> {
        let mut out: StridedArray<S::Reduced, E> = self.try_zeros()?;
        let mut out_iter = out.iter_mut_with_index();
        let idx: usize = idx[[]];
        while let Some((o, i)) = out_iter.next() {
            let mut j = 0;
            let mut reidx: S::Concrete = Default::default();
            for n in 0..S::NUM_DIMS {
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
    fn backward<const I: isize, S: Shape + ReduceShape<Axis<I>>>(
        &self,
        grad_inp: &mut Self::Storage<S, E>,
        idx: &Self::Storage<(), usize>,
        grad_out: &Self::Storage<S::Reduced, E>,
    ) -> Result<(), Self::Err> {
        let mut out_iter = grad_out.iter_with_index();
        let idx: usize = idx[[]];
        while let Some((o, i)) = out_iter.next() {
            let mut j = 0;
            let mut reidx: S::Concrete = Default::default();
            for n in 0..S::NUM_DIMS {
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

impl<E: Dtype> ReplaceAxisKernel<E> for Cpu {
    fn forward<const I: isize, D: Dim, S: Shape + ReplaceDim<I, D>>(
        &self,
        inp: &Self::Storage<S, E>,
        idx: &Self::Storage<(D,), usize>,
    ) -> Result<Self::Storage<S::Replaced, E>, Self::Err> {
        let mut out: StridedArray<S::Replaced, E> =
            self.try_zeros_like(inp.shape.replace(idx.shape.0))?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((o, mut i)) = out_iter.next() {
            let dim_i_idx = i[I as usize];
            i[I as usize] = idx[[dim_i_idx]];
            *o = inp[i];
        }
        Ok(out)
    }
    fn backward<const I: isize, D: Dim, S: Shape + ReplaceDim<I, D>>(
        &self,
        grad_inp: &mut Self::Storage<S, E>,
        idx: &Self::Storage<(D,), usize>,
        grad_out: &Self::Storage<S::Replaced, E>,
    ) -> Result<(), Self::Err> {
        let mut out_iter = grad_out.iter_with_index();
        while let Some((o, mut i)) = out_iter.next() {
            let dim_i_idx = i[I as usize];
            i[I as usize] = idx[[dim_i_idx]];
            grad_inp[i] += *o;
        }
        Ok(())
    }
}

// batched select (M, N) -> (B, Z, N)
impl<E: Dtype> SelectBatchKernel<E> for Cpu {
    fn forward<Batch: Dim, Seq: Dim, S1: Dim, S2: Dim>(
        &self,
        inp: &Self::Storage<(S1, S2), E>,
        idx: &Self::Storage<(Batch, Seq), usize>,
    ) -> Result<Self::Storage<(Batch, Seq, S2), E>, Self::Err> {
        let (_, s2) = inp.shape;
        let (batch, seq) = idx.shape;
        let mut out: StridedArray<(Batch, Seq, S2), E> = self.try_zeros_like((batch, seq, s2))?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((o, [b, s, i])) = out_iter.next() {
            let q = idx[[b, s]];
            *o = inp[[q, i]];
        }
        Ok(out)
    }
    fn backward<Batch: Dim, Seq: Dim, S1: Dim, S2: Dim>(
        &self,
        grad_inp: &mut Self::Storage<(S1, S2), E>,
        idx: &Self::Storage<(Batch, Seq), usize>,
        grad_out: &Self::Storage<(Batch, Seq, S2), E>,
    ) -> Result<(), Self::Err> {
        let mut out_iter = grad_out.iter_with_index();
        while let Some((o, [b, s, i])) = out_iter.next() {
            let q = idx[[b, s]];
            grad_inp[[q, i]] += *o;
        }
        Ok(())
    }
}
