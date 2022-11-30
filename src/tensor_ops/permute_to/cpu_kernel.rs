use crate::arrays::*;
use crate::tensor::cpu::{Cpu, StridedArray};

use super::PermuteKernel;

impl<E: Dtype> PermuteKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape<Concrete = Src::Concrete>, Ax: Axes>(
        &self,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: PermuteShapeTo<Dst, Ax>,
    {
        let out = inp.try_clone()?;
        let shape = inp.shape.permuted();
        let mut out: StridedArray<Dst, E> = StridedArray {
            data: out.data,
            shape,
            strides: out.strides,
        };
        for (i, idx) in Ax::as_array().into_iter().enumerate() {
            out.strides[i] = inp.strides[idx as usize];
        }
        Ok(out)
    }
    fn backward<Src: Shape, Dst: Shape<Concrete = Src::Concrete>, Ax: Axes>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: PermuteShapeTo<Dst, Ax>,
    {
        debug_assert_eq!(grad_inp.data.len(), grad_out.data.len());
        for (i, data_i) in grad_inp.buf_iter_mut().enumerate() {
            *data_i += grad_out.data[i];
        }
        Ok(())
    }
}
