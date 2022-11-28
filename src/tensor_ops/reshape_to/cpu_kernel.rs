use super::ReshapeKernel;
use crate::arrays::{Dtype, HasSameNumelAs, Shape};
use crate::devices::cpu::{Cpu, StridedArray};

impl<E: Dtype> ReshapeKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: HasSameNumelAs<Dst>,
    {
        Ok(StridedArray {
            data: inp.data.clone(),
            shape: dst,
            strides: dst.strides(),
        })
    }
    fn backward<Src: Shape, Dst: Shape>(
        &self,
        _inp: &Self::Storage<Src, E>,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: HasSameNumelAs<Dst>,
    {
        debug_assert_eq!(grad_inp.data.len(), grad_out.data.len());
        for (i, o) in grad_inp.buf_iter_mut().zip(grad_out.buf_iter()) {
            *i += *o;
        }
        Ok(())
    }
}
