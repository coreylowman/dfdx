use crate::{
    shapes::*,
    tensor::cpu::{Cpu, StridedArray},
};

impl<E: Dtype> super::BroadcastKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: BroadcastShapeTo<Dst, Ax>,
    {
        Ok(StridedArray {
            data: inp.data.clone(),
            shape: dst,
            strides: inp.shape.broadcast_strides(inp.strides),
        })
    }

    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: BroadcastShapeTo<Dst, Ax>,
    {
        debug_assert_eq!(grad_out.data.len(), grad_inp.data.len());
        for (i, o) in grad_inp.buf_iter_mut().zip(grad_out.buf_iter()) {
            *i += *o;
        }
        Ok(())
    }
}
