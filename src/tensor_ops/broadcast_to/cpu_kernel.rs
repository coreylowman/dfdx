use crate::{
    shapes::*,
    tensor::{Cpu, Tensor},
    unique_id::unique_id,
};

impl<E: Dtype> super::BroadcastKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Tensor<Src, E, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err>
    where
        Src: BroadcastShapeTo<Dst, Ax>,
    {
        Ok(Tensor {
            id: unique_id(),
            data: inp.data.clone(),
            shape: dst,
            strides: inp.shape.broadcast_strides(inp.strides),
            device: self.clone(),
            tape: Default::default(),
        })
    }

    fn backward(
        &self,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(grad_out.len(), grad_inp.len());
        for (i, o) in grad_inp.iter_mut().zip(grad_out.iter()) {
            *i += *o;
        }
        Ok(())
    }
}
