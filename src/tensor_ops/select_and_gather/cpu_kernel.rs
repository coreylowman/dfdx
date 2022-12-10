#![allow(clippy::needless_range_loop)]

use crate::shapes::{Axes, Dtype, RemoveDimTo, ReplaceDimTo, Shape};
use crate::tensor::cpu::{Cpu, LendingIterator, StridedArray};

impl<E: Dtype> super::ReplaceDimKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Self::Storage<Src, E>,
        idx: &Self::Storage<Idx, usize>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: ReplaceDimTo<Dst, Idx>,
    {
        let ax = Src::Ax::as_array()[0] as usize;
        assert!(<Idx as Shape>::NUM_DIMS >= ax);

        let offset = <Idx as Shape>::NUM_DIMS - ax;

        let mut out = StridedArray::new(inp.shape.replace(idx.shape))?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((x, i_replaced)) = out_iter.next() {
            let mut i_idx: <Idx as Shape>::Concrete = Default::default();
            let mut i_inp: Src::Concrete = Default::default();

            // # Construct the `idx` indices
            // the indices into `idx` will always share the same "head"
            // as the indices for `out`.
            for j in 0..<Idx as Shape>::NUM_DIMS {
                i_idx[j] = i_replaced[j];
            }

            // # Construct the `inp` indices
            for j in 0..Src::NUM_DIMS {
                i_inp[j] = match j.cmp(&ax) {
                    std::cmp::Ordering::Less => i_replaced[j],
                    std::cmp::Ordering::Equal => idx[i_idx],
                    std::cmp::Ordering::Greater => i_replaced[j - 1 + offset],
                };
            }
            *x = inp[i_inp];
        }
        Ok(out)
    }

    fn backward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        idx: &Self::Storage<Idx, usize>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReplaceDimTo<Dst, Idx>,
    {
        let ax = Src::Ax::as_array()[0] as usize;

        // NOTE: this is the same exact indexing logic as forward
        let offset = <Idx as Shape>::NUM_DIMS - ax;

        let mut out_iter = grad_out.iter_with_index();
        while let Some((x, i_replaced)) = out_iter.next() {
            let mut i_idx: <Idx as Shape>::Concrete = Default::default();
            let mut i_inp: Src::Concrete = Default::default();
            for j in 0..<Idx as Shape>::NUM_DIMS {
                i_idx[j] = i_replaced[j];
            }
            for j in 0..Src::NUM_DIMS {
                i_inp[j] = match j.cmp(&ax) {
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

impl<E: Dtype> super::RemoveDimKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Self::Storage<Src, E>,
        idx: &Self::Storage<Idx, usize>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: RemoveDimTo<Dst, Idx>,
    {
        let ax = Src::Ax::as_array()[0] as usize;

        let mut out = StridedArray::new(inp.shape.remove(idx.shape))?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((x, i_replaced)) = out_iter.next() {
            let mut i_idx: <Idx as Shape>::Concrete = Default::default();
            let mut i_inp: Src::Concrete = Default::default();

            // # Construct the `idx` indices
            // the indices into `idx` will always share the same "head"
            // as the indices for `out`.
            for j in 0..<Idx as Shape>::NUM_DIMS {
                i_idx[j] = i_replaced[j];
            }

            // # Construct the `inp` indices
            for j in 0..Src::NUM_DIMS {
                i_inp[j] = match j.cmp(&ax) {
                    std::cmp::Ordering::Less => i_replaced[j],
                    std::cmp::Ordering::Equal => idx[i_idx],
                    std::cmp::Ordering::Greater => i_replaced[j - 1],
                };
            }
            *x = inp[i_inp];
        }
        Ok(out)
    }

    fn backward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        idx: &Self::Storage<Idx, usize>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: RemoveDimTo<Dst, Idx>,
    {
        let ax = Src::Ax::as_array()[0] as usize;

        let mut out_iter = grad_out.iter_with_index();
        while let Some((x, i_replaced)) = out_iter.next() {
            let mut i_idx: <Idx as Shape>::Concrete = Default::default();
            let mut i_inp: Src::Concrete = Default::default();
            for j in 0..<Idx as Shape>::NUM_DIMS {
                i_idx[j] = i_replaced[j];
            }
            for j in 0..Src::NUM_DIMS {
                i_inp[j] = match j.cmp(&ax) {
                    std::cmp::Ordering::Less => i_replaced[j],
                    std::cmp::Ordering::Equal => idx[i_idx],
                    std::cmp::Ordering::Greater => i_replaced[j - 1],
                };
            }
            grad_inp[i_inp] += *x;
        }
        Ok(())
    }
}
