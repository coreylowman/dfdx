#![allow(clippy::needless_range_loop)]
/// # Implementation details
/// ReplaceDimKernel handles two cases:
/// 1. Removing a dim completely
/// 2. Replacing a dim with a new dim
///
/// These two cases only differ by the shape of the index,
/// and thus how you create the index into the input.
///
///
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

        // NOTE: this is how the difference in cases is detected.
        // For removing a dim, offset will be 0
        // For replacing a dim, offset will be 1
        // this is used in the innermost loop below to change the value pulled from
        // `i_replaced`
        let replacing = <Src::Index as Shape>::NUM_DIMS - I as usize;

        let mut out = StridedArray::new(inp.shape.replace(idx.shape))?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((x, i_replaced)) = out_iter.next() {
            let mut i_idx: <Src::Index as Shape>::Concrete = Default::default();
            let mut i_inp: Src::Concrete = Default::default();

            // # Construct the `idx` indices
            // the indices into `idx` will always share the same "head"
            // as the indices for `out`.
            for j in 0..<Src::Index as Shape>::NUM_DIMS {
                i_idx[j] = i_replaced[j];
            }

            // # Construct the `inp` indices
            for j in 0..Src::NUM_DIMS {
                i_inp[j] = match j.cmp(&(I as usize)) {
                    std::cmp::Ordering::Less => i_replaced[j],
                    std::cmp::Ordering::Equal => idx[i_idx],
                    std::cmp::Ordering::Greater => i_replaced[j - 1 + replacing],
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
        // NOTE: this is the same exact indexing logic as forward
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
        let (_s1, s2) = inp.shape;
        let (batch, seq) = idx.shape;
        let mut out = StridedArray::new((batch, seq, s2))?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((o, [i_batch, i_seq, i_s2])) = out_iter.next() {
            let i_s1 = idx[[i_batch, i_seq]];
            *o = inp[[i_s1, i_s2]];
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
        while let Some((o, [i_batch, i_seq, i_s2])) = out_iter.next() {
            let i_s1 = idx[[i_batch, i_seq]];
            grad_inp[[i_s1, i_s2]] += *o;
        }
        Ok(())
    }
}
