use crate::arrays::*;
use crate::devices::cpu::{Cpu, StridedArray};

use super::PermuteKernel;

impl<E: Dtype> PermuteKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape<Concrete = Src::Concrete>, Axes: AxesAsArray>(
        &self,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: PermuteShapeTo<Dst, Axes>,
    {
        let out = inp.try_clone()?;
        let shape = inp.shape.permuted();
        let mut out: StridedArray<Dst, E> = StridedArray {
            data: out.data,
            shape,
            strides: StridesFor(out.strides.0),
        };
        for (i, idx) in Axes::as_array().into_iter().enumerate() {
            out.strides.0[i] = inp.strides.0[idx as usize];
        }
        Ok(out)
    }
    fn backward<Src: Shape, Dst: Shape<Concrete = Src::Concrete>, Axes: AxesAsArray>(
        &self,
        _inp: &Self::Storage<Src, E>,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: PermuteShapeTo<Dst, Axes>,
    {
        debug_assert_eq!(grad_inp.data.len(), grad_out.data.len());
        for (i, data_i) in grad_inp.buf_iter_mut().enumerate() {
            *data_i += grad_out.data[i];
        }
        Ok(())
    }
}
