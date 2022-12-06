#![allow(clippy::needless_range_loop)]
/// # Implementation details
/// ReplaceDimKernel handles three cases:
/// 1. Removing a dim completely
/// 2. Replacing a dim with a new dim
/// 3. Batch select (adding a dim)
///
/// These three cases only differ by the shape of the index,
/// and thus how you create the index into the input.
use crate::shapes::{Axes, Dtype, ReplaceDimTo, Shape};
use crate::tensor::cpu::{Cpu, LendingIterator, StridedArray};

use super::ReplaceDimKernel;

impl<E: Dtype> ReplaceDimKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes<Array = [isize; 1]>>(
        &self,
        inp: &Self::Storage<Src, E>,
        idx: &Self::Storage<Src::Index, usize>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: ReplaceDimTo<Dst, Ax>,
    {
        let ax = Ax::as_array()[0] as usize;
        assert!(<Src::Index as Shape>::NUM_DIMS >= ax);

        // NOTE: this is how the difference in cases is detected.
        // For removing a dim, offset will be 0
        // For replacing a dim, offset will be 1
        // this is used in the innermost loop below to change the value pulled from
        // `i_replaced`
        let replacing = <Src::Index as Shape>::NUM_DIMS - ax;

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
                i_inp[j] = match j.cmp(&ax) {
                    std::cmp::Ordering::Less => i_replaced[j],
                    std::cmp::Ordering::Equal => idx[i_idx],
                    std::cmp::Ordering::Greater => i_replaced[j - 1 + replacing],
                };
            }
            *x = inp[i_inp];
        }
        Ok(out)
    }

    fn backward<Src: Shape, Dst: Shape, Ax: Axes<Array = [isize; 1]>>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        idx: &Self::Storage<Src::Index, usize>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReplaceDimTo<Dst, Ax>,
    {
        let ax = Ax::as_array()[0] as usize;

        // NOTE: this is the same exact indexing logic as forward
        let offset = <Src::Index as Shape>::NUM_DIMS - ax;

        let mut out_iter = grad_out.iter_with_index();
        while let Some((x, i_replaced)) = out_iter.next() {
            let mut i_idx: <Src::Index as Shape>::Concrete = Default::default();
            let mut i_inp: Src::Concrete = Default::default();
            for j in 0..<Src::Index as Shape>::NUM_DIMS {
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
