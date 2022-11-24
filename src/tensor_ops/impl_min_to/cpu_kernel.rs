use crate::{
    arrays::{BroadcastStrides, Shape},
    devices::{cpu::StridedArray, Cpu, FullUnaryKernel},
};

use super::MinReduceKernelOp;

impl<const N: usize, Axes, Src, Dst> FullUnaryKernel<MinReduceKernelOp<Axes>, Src, Dst, f32> for Cpu
where
    Dst: BroadcastStrides<Src, Axes>,
    Src: Shape<Concrete = [usize; N]>,
    Dst: Shape + Default,
{
    fn unary_fwd(
        &self,
        _op: MinReduceKernelOp<Axes>,
        inp: &Self::Storage<Src, f32>,
    ) -> Result<Self::Storage<Dst, f32>, Self::Err> {
        let mut out: StridedArray<Dst, f32> =
            StridedArray::try_new_with(Dst::default(), f32::INFINITY)?;
        let mut out_iter = out.iter_mut_as(&inp.shape);
        let mut inp_iter = inp.iter();
        while let Some((out_i, inp_i)) = out_iter.next().zip(inp_iter.next()) {
            *out_i = f32::min(*out_i, *inp_i);
        }
        Ok(out)
    }

    fn unary_bwd(
        &self,
        _op: MinReduceKernelOp<Axes>,
        inp: &Self::Storage<Src, f32>,
        grad_inp: &mut Self::Storage<Src, f32>,
        out: &Self::Storage<Dst, f32>,
        grad_out: &Self::Storage<Dst, f32>,
    ) -> Result<(), Self::Err> {
        let mut inp_iter = inp.iter();
        let mut grad_inp_itr = grad_inp.iter_mut();
        let mut out_iter = out.iter_as(&inp.shape);
        let mut grad_out_iter = grad_out.iter_as(&inp.shape);
        for _ in 0..inp.shape.num_elements() {
            let d = if out_iter.next().unwrap() == inp_iter.next().unwrap() {
                1.0
            } else {
                0.0
            };
            *grad_inp_itr.next().unwrap() += *grad_out_iter.next().unwrap() * d;
        }
        Ok(())
    }
}
