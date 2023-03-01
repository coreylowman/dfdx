use crate::shapes::*;
use crate::tensor::{cpu::Cpu, Tensor};
use crate::unique_id::unique_id;

impl<E: Dtype> super::PermuteKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        inp: &Tensor<Src, E, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err>
    where
        Src: PermuteShapeTo<Dst, Ax>,
    {
        Ok(Tensor {
            id: unique_id(),
            data: inp.data.clone(),
            shape: inp.shape.permuted(),
            strides: inp.shape.permute_strides(inp.strides),
            device: self.clone(),
            tape: Default::default(),
        })
    }
    fn backward(
        &self,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(grad_inp.len(), grad_out.len());
        for (i, o) in grad_inp.iter_mut().zip(grad_out.iter()) {
            *i += *o;
        }
        Ok(())
    }
}
