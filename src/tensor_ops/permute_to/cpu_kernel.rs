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
        let mut out: StridedArray<Dst, E> = StridedArray {
            data: inp.data.clone(),
            shape: inp.shape.permuted(),
            strides: inp.strides,
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
        for (inp_i, out_i) in grad_inp.buf_iter_mut().zip(grad_out.buf_iter()) {
            *inp_i += *out_i;
        }
        Ok(())
    }
}
