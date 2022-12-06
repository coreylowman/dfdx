#![allow(clippy::needless_range_loop)]

use crate::shapes::{Dim, Dtype, ReplaceDimTo, Shape};
use crate::tensor::cpu::{Cpu, LendingIterator, StridedArray};

use super::{ReplaceDimKernel, SelectBatchKernel};

impl<E: Dtype> ReplaceDimKernel<E> for Cpu {
    fn forward<const I: isize, Src: Shape, Dst: Shape>(
        &self,
        inp: &Self::Storage<Src, E>,
        idx: &Self::Storage<Src::Index, usize>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: ReplaceDimTo<Dst, I>,
    {
        assert!(<Src::Index as Shape>::NUM_DIMS >= I as usize);
        let offset = <Src::Index as Shape>::NUM_DIMS - I as usize;
        let mut out = StridedArray::new(inp.shape.replace(idx.shape))?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((x, i_replaced)) = out_iter.next() {
            let mut i_idx: <Src::Index as Shape>::Concrete = Default::default();
            let mut i_inp: Src::Concrete = Default::default();
            for j in 0..<Src::Index as Shape>::NUM_DIMS {
                i_idx[j] = i_replaced[j];
            }
            for j in 0..Src::NUM_DIMS {
                i_inp[j] = match j.cmp(&(I as usize)) {
                    std::cmp::Ordering::Less => i_replaced[j],
                    std::cmp::Ordering::Equal => idx[i_idx],
                    std::cmp::Ordering::Greater => i_replaced[j - 1 + offset],
                };
            }
            *x = inp[i_inp];
        }
        Ok(out)
    }
    fn backward<const I: isize, Src: Shape, Dst: Shape>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        idx: &Self::Storage<Src::Index, usize>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReplaceDimTo<Dst, I>,
    {
        let offset = <Src::Index as Shape>::NUM_DIMS - I as usize;
        let mut out_iter = grad_out.iter_with_index();
        while let Some((x, i_replaced)) = out_iter.next() {
            let mut i_idx: <Src::Index as Shape>::Concrete = Default::default();
            let mut i_inp: Src::Concrete = Default::default();
            for j in 0..<Src::Index as Shape>::NUM_DIMS {
                i_idx[j] = i_replaced[j];
            }
            for j in 0..Src::NUM_DIMS {
                i_inp[j] = match j.cmp(&(I as usize)) {
                    std::cmp::Ordering::Less => i_replaced[j],
                    std::cmp::Ordering::Equal => idx[i_idx],
                    std::cmp::Ordering::Greater => i_replaced[j - 1 + offset],
                };
            }
            grad_inp[i_inp] += *x;
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
        let mut out = StridedArray::new((batch, seq, s2))?;
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
