use crate::arrays::*;
use crate::devices::cpu::{Cpu, LendingIterator, StridedArray};
use crate::devices::{device::*, unary_ops};

impl<
        const N: usize,
        Axes,
        Src: Shape<Concrete = [usize; N]>,
        Dst: Shape + Default,
        Elem: Dtype + for<'a> std::ops::AddAssign<&'a Elem>,
    > UnaryKernel<unary_ops::Sum<Axes>, Src, Dst, Elem> for Cpu
where
    Dst: BroadcastStrides<Src, Axes>,
{
    fn unary_fwd(
        &self,
        _op: unary_ops::Sum<Axes>,
        inp: &Self::Storage<Src, Elem>,
    ) -> Result<Self::Storage<Dst, Elem>, Self::Err> {
        let mut out: StridedArray<Dst, Elem> = self.try_zeros()?;
        let mut out_iter = out.iter_mut_as(&inp.shape);
        let mut inp_iter = inp.iter();
        while let Some((o, i)) = out_iter.next().zip(inp_iter.next()) {
            o.add_assign(i);
        }
        Ok(out)
    }

    fn unary_bwd(
        &self,
        _op: unary_ops::Sum<Axes>,
        inp: &Self::Storage<Src, Elem>,
        grad_inp: &mut Self::Storage<Src, Elem>,
        grad_out: &Self::Storage<Dst, Elem>,
    ) -> Result<(), Self::Err> {
        let mut inp_iter = grad_inp.iter_mut();
        let mut out_iter = grad_out.iter_as(&inp.shape);
        while let Some((i, o)) = inp_iter.next().zip(out_iter.next()) {
            i.add_assign(o);
        }
        Ok(())
    }
}

impl<const N: usize, Axes, Src: Shape<Concrete = [usize; N]>, Dst: Shape + Default>
    FullUnaryKernel<unary_ops::MaxReduce<Axes>, Src, Dst, f32> for Cpu
where
    Dst: BroadcastStrides<Src, Axes>,
{
    fn unary_fwd(
        &self,
        _op: unary_ops::MaxReduce<Axes>,
        inp: &Self::Storage<Src, f32>,
    ) -> Result<Self::Storage<Dst, f32>, Self::Err> {
        let mut out: StridedArray<Dst, f32> =
            StridedArray::try_new_with(Dst::default(), f32::NEG_INFINITY)?;
        let mut out_iter = out.iter_mut_as(&inp.shape);
        let mut inp_iter = inp.iter();
        while let Some((out_i, inp_i)) = out_iter.next().zip(inp_iter.next()) {
            *out_i = f32::max(*out_i, *inp_i);
        }
        Ok(out)
    }

    fn unary_bwd(
        &self,
        _op: unary_ops::MaxReduce<Axes>,
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

impl<const N: usize, Axes, Src: Shape<Concrete = [usize; N]>, Dst: Shape + Default>
    FullUnaryKernel<unary_ops::MinReduce<Axes>, Src, Dst, f32> for Cpu
where
    Dst: BroadcastStrides<Src, Axes>,
{
    fn unary_fwd(
        &self,
        _op: unary_ops::MinReduce<Axes>,
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
        _op: unary_ops::MinReduce<Axes>,
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
