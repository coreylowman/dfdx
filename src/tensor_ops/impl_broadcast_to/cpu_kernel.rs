use super::BroadcastKernelOp;
use crate::arrays::*;
use crate::devices::cpu::{Cpu, StridedArray};
use crate::devices::UnaryKernel;

impl<Axes, Src: Shape, Dst: Shape, E: Dtype> UnaryKernel<BroadcastKernelOp<Dst, Axes>, Src, Dst, E>
    for Cpu
where
    Src: BroadcastStrides<Dst, Axes>,
{
    fn unary_fwd(
        &self,
        op: BroadcastKernelOp<Dst, Axes>,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err> {
        let strides: StridesFor<Dst> = inp.shape.broadcast_strides(inp.strides);
        let out: StridedArray<Dst, E> = StridedArray {
            data: inp.data.clone(),
            shape: op.0,
            strides,
        };
        Ok(out)
    }

    fn unary_bwd(
        &self,
        _op: BroadcastKernelOp<Dst, Axes>,
        _inp: &Self::Storage<Src, E>,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(grad_out.data.len(), grad_inp.data.len());
        for (i, data_i) in grad_inp.buf_iter_mut().enumerate() {
            *data_i += grad_out.data[i];
        }
        Ok(())
    }
}
