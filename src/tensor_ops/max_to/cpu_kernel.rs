use crate::{
    shapes::{Axes, Dtype, HasAxes, ReduceShapeTo, Shape},
    tensor::{Cpu, Tensor, ZerosTensor},
    tensor_ops::utilities::reduction_utils::index_for_reductions,
};

use num_traits::Float;

impl<E: Dtype + Float> super::MaxReduceKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Tensor<Src, E, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let mut out = self.try_zeros_like(&dst)?;
        if Dst::NUM_DIMS == 0 {
            debug_assert_eq!(out.data.len(), 1);
            let mut tmp: E = E::neg_infinity();
            for i in inp.buf_iter() {
                tmp = i.max(tmp);
            }
            std::sync::Arc::get_mut(&mut out.data).unwrap()[0] = tmp;
        } else {
            let num_elems_reduced = <Src as HasAxes<Ax>>::size(&inp.shape);
            let inp_buf = inp.data.as_ref();
            let mut idx = index_for_reductions::<Src, Ax>(inp.shape, inp.strides);
            for o in out.buf_iter_mut() {
                let mut tmp: E = E::neg_infinity();
                for _ in 0..num_elems_reduced {
                    tmp = tmp.max(inp_buf[idx.next().unwrap()]);
                }
                *o = tmp;
            }
        }
        Ok(out)
    }

    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<Dst, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let num_elems_reduced = <Src as HasAxes<Ax>>::size(&inp.shape);

        let inp_buf = inp.data.as_ref();
        let mut inp_idx = index_for_reductions::<Src, Ax>(inp.shape, inp.strides);

        for (&o, &go) in out.buf_iter().zip(grad_out.iter()) {
            for _ in 0..num_elems_reduced {
                let inp_i = inp_idx.next().unwrap();
                let d = if o == inp_buf[inp_i] {
                    E::one()
                } else {
                    E::zero()
                };
                grad_inp[inp_i] += go * d;
            }
        }
        Ok(())
    }
}
