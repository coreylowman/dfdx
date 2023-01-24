use crate::shapes::*;
use crate::tensor::cpu::{Cpu, StridedArray};

impl<E: Dtype> super::PermuteKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: PermuteShapeTo<Dst, Ax>,
    {
        Ok(StridedArray {
            data: inp.data.clone(),
            shape: inp.shape.permuted(),
            strides: inp.shape.permute_strides(inp.strides),
        })
    }
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: PermuteShapeTo<Dst, Ax>,
    {
        debug_assert_eq!(grad_inp.data.len(), grad_out.data.len());
        for (i, o) in grad_inp.buf_iter_mut().zip(grad_out.buf_iter()) {
            *i += *o;
        }
        Ok(())
    }
}
