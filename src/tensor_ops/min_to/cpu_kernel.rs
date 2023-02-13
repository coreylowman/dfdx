use crate::{
    shapes::{Axes, Dtype, ReduceShapeTo, Shape},
    tensor::cpu::{Cpu, LendingIterator, StridedArray},
};

use num_traits::Float;

impl<F: Float + Dtype> super::MinReduceKernel<F> for Cpu {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, F>,
    ) -> Result<Self::Storage<Dst, F>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let mut out: StridedArray<Dst, F> = StridedArray::try_new_with(dst, F::infinity())?;
        let mut out_iter = out.iter_mut_as(&inp.shape);
        let mut inp_iter = inp.iter();
        while let Some((out_i, inp_i)) = out_iter.next().zip(inp_iter.next()) {
            *out_i = F::min(*out_i, *inp_i);
        }
        Ok(out)
    }

    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        inp: &Self::Storage<Src, F>,
        grad_inp: &mut Self::Storage<Src, F>,
        out: &Self::Storage<Dst, F>,
        grad_out: &Self::Storage<Dst, F>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let mut inp_iter = inp.iter();
        let mut grad_inp_itr = grad_inp.iter_mut();
        let mut out_iter = out.iter_as(&inp.shape);
        let mut grad_out_iter = grad_out.iter_as(&inp.shape);
        for _ in 0..inp.shape.num_elements() {
            let d = if out_iter.next().unwrap() == inp_iter.next().unwrap() {
                F::one()
            } else {
                F::zero()
            };
            *grad_inp_itr.next().unwrap() += *grad_out_iter.next().unwrap() * d;
        }
        Ok(())
    }
}
