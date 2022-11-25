use crate::arrays::*;
use crate::devices::cpu::{Cpu, StridedArray};
use crate::tensor_ops::ops::UnaryKernel;

use super::PermuteKernelOp;

impl<
        const N: usize,
        Axes: AxesAsArray<Array = [isize; N]>,
        Src: Shape<Concrete = [usize; N]>,
        Dst: Shape<Concrete = [usize; N]>,
    > UnaryKernel<PermuteKernelOp<Dst, Axes>, Src, Dst, f32> for Cpu
where
    Src: PermuteShapeTo<Dst, Axes>,
{
    fn unary_fwd(
        &self,
        op: PermuteKernelOp<Dst, Axes>,
        inp: &Self::Storage<Src, f32>,
    ) -> Result<Self::Storage<Dst, f32>, Self::Err> {
        let out = inp.try_clone()?;
        let mut out: StridedArray<Dst, f32> = StridedArray {
            data: out.data,
            shape: op.0,
            strides: StridesFor(out.strides.0),
        };
        let reidx = Axes::as_array().map(|x| x as usize);
        for (i, &idx) in reidx.iter().enumerate() {
            out.strides.0[i] = inp.strides.0[idx];
        }
        Ok(out)
    }

    fn unary_bwd(
        &self,
        _op: PermuteKernelOp<Dst, Axes>,
        _inp: &Self::Storage<Src, f32>,
        grad_inp: &mut Self::Storage<Src, f32>,
        grad_out: &Self::Storage<Dst, f32>,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(grad_inp.data.len(), grad_out.data.len());
        for (i, data_i) in grad_inp.buf_iter_mut().enumerate() {
            *data_i += grad_out.data[i];
        }
        Ok(())
    }
}
