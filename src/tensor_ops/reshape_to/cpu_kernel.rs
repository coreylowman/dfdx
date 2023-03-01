use crate::shapes::{Dtype, Shape};
use crate::tensor::{
    cpu::{LendingIterator, NdIndex},
    Cpu, Tensor, ZerosTensor,
};

impl<E: Dtype> super::ReshapeKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape>(
        &self,
        dst: &Dst,
        inp: &Tensor<Src, E, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err> {
        let mut out = self.try_zeros_like(dst)?;
        let mut inp_iter = inp.iter();
        let mut out_iter = out.iter_mut();
        while let Some((o, i)) = out_iter.next().zip(inp_iter.next()) {
            *o = *i;
        }
        Ok(out)
    }
    fn backward<Src: Shape, Dst: Shape>(
        &self,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<Dst, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let mut inp_idx = NdIndex::new(inp.shape, inp.strides);
        let mut out_idx = NdIndex::new(out.shape, out.strides);
        while let Some((i, o)) = inp_idx.next().zip(out_idx.next()) {
            grad_inp[i] += grad_out[o];
        }
        Ok(())
    }
}
