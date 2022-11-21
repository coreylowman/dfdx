use crate::arrays::*;
use crate::devices::cpu::{Cpu, StridedArray};
use crate::devices::{device::*, unary_ops};

impl<Axes, Src: Shape, Dst: Shape + Default, Elem: Dtype + std::ops::AddAssign<Elem>>
    UnaryKernel<unary_ops::Broadcast<Dst, Axes>, Src, Dst, Elem> for Cpu
where
    Src: BroadcastStrides<Dst, Axes>,
{
    fn unary_fwd(
        &self,
        op: unary_ops::Broadcast<Dst, Axes>,
        inp: &Self::Storage<Src, Elem>,
    ) -> Result<Self::Storage<Dst, Elem>, Self::Err> {
        let strides: StridesFor<Dst> = inp.shape.broadcast_strides(inp.strides);
        let out: StridedArray<Dst, Elem> = StridedArray {
            data: inp.data.clone(),
            shape: op.0,
            strides,
        };
        Ok(out)
    }
    fn unary_bwd(
        &self,
        _op: unary_ops::Broadcast<Dst, Axes>,
        _inp: &Self::Storage<Src, Elem>,
        grad_inp: &mut Self::Storage<Src, Elem>,
        grad_out: &Self::Storage<Dst, Elem>,
    ) {
        assert_eq!(grad_out.data.len(), grad_inp.data.len());
        let data = std::sync::Arc::make_mut(&mut grad_inp.data);
        for (i, data_i) in data.iter_mut().enumerate() {
            *data_i += grad_out.data[i];
        }
    }
}
