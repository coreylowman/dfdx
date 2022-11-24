use super::ReshapeKernelOp;
use crate::arrays::{Dtype, HasSameNumelAs, Shape};
use crate::devices::cpu::{Cpu, StridedArray};
use crate::tensor_ops::utils::UnaryKernel;

impl<Src: Shape, Dst: Shape, E: Dtype + std::ops::AddAssign>
    UnaryKernel<ReshapeKernelOp<Dst>, Src, Dst, E> for Cpu
where
    Src: HasSameNumelAs<Dst>,
{
    fn unary_fwd(
        &self,
        op: ReshapeKernelOp<Dst>,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err> {
        Ok(StridedArray {
            data: inp.data.clone(),
            shape: op.0,
            strides: op.0.strides(),
        })
    }
    fn unary_bwd(
        &self,
        _op: ReshapeKernelOp<Dst>,
        _inp: &Self::Storage<Src, E>,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(grad_inp.data.len(), grad_out.data.len());
        for (i, o) in grad_inp.buf_iter_mut().zip(grad_out.buf_iter()) {
            *i += *o;
        }
        Ok(())
    }
}
