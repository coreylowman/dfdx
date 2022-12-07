use crate::{
    shapes::{Axes, Dtype, ReduceShapeTo, Shape},
    tensor::cpu::{Cpu, LendingIterator, StridedArray},
};

use super::SumKernel;

impl<E: Dtype> SumKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let mut out: StridedArray<Dst, E> = StridedArray::new(dst)?;
        let mut out_iter = out.iter_mut_as(&inp.shape);
        let mut inp_iter = inp.iter();
        while let Some((o, i)) = out_iter.next().zip(inp_iter.next()) {
            o.add_assign(*i);
        }
        Ok(out)
    }
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let mut out_iter = grad_out.iter_as(&grad_inp.shape);
        let mut inp_iter = grad_inp.iter_mut();
        while let Some((i, o)) = inp_iter.next().zip(out_iter.next()) {
            i.add_assign(*o);
        }
        Ok(())
    }
}
