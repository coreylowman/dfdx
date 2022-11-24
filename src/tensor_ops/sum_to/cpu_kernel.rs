use crate::{
    arrays::{BroadcastStrides, Dtype, Shape},
    devices::{
        cpu::{Cpu, LendingIterator, StridedArray},
        Zeros,
    },
    tensor_ops::utils::UnaryKernel,
};

use super::SumKernelOp;

impl<const N: usize, Axes, Src: Shape<Concrete = [usize; N]>, Dst: Shape + Default, E: Dtype>
    UnaryKernel<SumKernelOp<Axes>, Src, Dst, E> for Cpu
where
    Dst: BroadcastStrides<Src, Axes>,
{
    fn unary_fwd(
        &self,
        _op: SumKernelOp<Axes>,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err> {
        let mut out: StridedArray<Dst, E> = self.try_zeros()?;
        let mut out_iter = out.iter_mut_as(&inp.shape);
        let mut inp_iter = inp.iter();
        while let Some((o, i)) = out_iter.next().zip(inp_iter.next()) {
            o.add_assign(*i);
        }
        Ok(out)
    }

    fn unary_bwd(
        &self,
        _op: SumKernelOp<Axes>,
        inp: &Self::Storage<Src, E>,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err> {
        let mut inp_iter = grad_inp.iter_mut();
        let mut out_iter = grad_out.iter_as(&inp.shape);
        while let Some((i, o)) = inp_iter.next().zip(out_iter.next()) {
            i.add_assign(*o);
        }
        Ok(())
    }
}
