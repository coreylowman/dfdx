use super::BroadcastKernel;
use crate::arrays::*;
use crate::devices::cpu::{Cpu, StridedArray};

impl<E: Dtype> BroadcastKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: BroadcastShapeTo<Dst, Ax>,
    {
        let strides: StridesFor<Dst> = inp.shape.broadcast_strides(inp.strides);
        let out: StridedArray<Dst, E> = StridedArray {
            data: inp.data.clone(),
            shape: dst,
            strides,
        };
        Ok(out)
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
        for (i, data_i) in grad_inp.buf_iter_mut().enumerate() {
            *data_i += grad_out.data[i];
        }
        Ok(())
    }
}
