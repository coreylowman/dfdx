use crate::shapes::{Dtype, HasSameNumelAs, Shape};
use crate::tensor::cpu::{Cpu, LendingIterator, StridedArray};

impl<E: Dtype> super::ReshapeKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: HasSameNumelAs<Dst>,
    {
        let mut out = StridedArray::new(dst)?;
        let mut inp_iter = inp.iter();
        let mut out_iter = out.iter_mut();
        while let Some((o, i)) = out_iter.next().zip(inp_iter.next()) {
            *o = *i;
        }
        Ok(out)
    }

    fn backward<Src: Shape, Dst: Shape>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: HasSameNumelAs<Dst>,
    {
        let mut inp_iter = grad_inp.iter_mut();
        let mut out_iter = grad_out.iter();
        while let Some((i, o)) = inp_iter.next().zip(out_iter.next()) {
            *i += *o;
        }
        Ok(())
    }
}
