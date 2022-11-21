use crate::arrays::{Dtype, HasSameNumelAs, Shape};
use crate::devices::cpu::device::StridedArray;
use crate::devices::device::UnaryKernel;
use crate::devices::unary_ops;

use super::Cpu;

impl<Src: Shape, Dst: Shape, E: Dtype + std::ops::AddAssign>
    UnaryKernel<unary_ops::Reshape<Dst>, Src, Dst, E> for Cpu
where
    Src: HasSameNumelAs<Dst>,
{
    fn unary_fwd(
        &self,
        op: unary_ops::Reshape<Dst>,
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
        op: unary_ops::Reshape<Dst>,
        _inp: &Self::Storage<Src, E>,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) {
        assert_eq!(grad_inp.data.len(), grad_out.data.len());
        let inp_data = std::sync::Arc::make_mut(&mut grad_inp.data);
        for (i, o) in inp_data.iter_mut().zip(grad_out.data.iter()) {
            *i += *o;
        }
    }
}
