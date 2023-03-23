use crate::prelude::cpu::{LendingIterator, NdIndex};

use super::*;

impl<E: Unit> SliceKernel<E> for Cpu {
    fn forward<Src: Shape + SliceShape<Slice>, Slice>(
        &self,
        inp: &Tensor<Src, E, Self>,
        slice: &Slice,
    ) -> Result<Tensor<Src::Sliced, E, Self>, Self::Err> {
        let dst = inp.shape.slice(slice).unwrap();
        let mut out = self.try_zeros_like(&dst)?;

        let mut inp_idx = NdIndex::new(dst, inp.strides);
        let mut out_iter = out.iter_mut();

        let start_idx = NdIndex::new(inp.shape, inp.strides)
            .get_strided_index(inp.shape.first_idx_in_slice(slice));
        let view = &inp.data[start_idx..];

        println!("{} {}", start_idx, inp.shape.first_idx_in_slice(slice));

        while let Some((inp_i, o)) = inp_idx.next().zip(out_iter.next()) {
            *o = view[inp_i];
        }

        Ok(out)
    }

    fn backward<Src: Shape + SliceShape<Slice>, Slice>(
        &self,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut Vec<E>,
        grad_out: &Vec<E>,
        slice: &Slice,
    ) -> Result<(), Self::Err> {
        let dst = inp.shape.slice(slice).unwrap();

        let mut inp_idx = NdIndex::new(dst, inp.strides);
        let mut out_iter = grad_out.iter();

        let start_idx = NdIndex::new(inp.shape, inp.strides)
            .get_strided_index(inp.shape.first_idx_in_slice(slice));
        let view = &mut grad_inp[start_idx..];

        while let Some((inp_i, o)) = inp_idx.next().zip(out_iter.next()) {
            view[inp_i] = *o;
        }

        Ok(())
    }
}
