use crate::{
    shapes::{Axes, Dtype, HasAxes, ReduceShapeTo, Shape},
    tensor::cpu::{Cpu, StridedArray},
    tensor_ops::utilities::reduction_utils::index_for_reductions,
};

use num_traits::Float;

impl<E: Dtype + Float> super::MinReduceKernel<E> for Cpu {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let mut out: StridedArray<Dst, E> = StridedArray::new(dst)?;
        if Dst::NUM_DIMS == 0 {
            debug_assert_eq!(out.data.len(), 1);
            let mut tmp: E = E::infinity();
            for i in inp.buf_iter() {
                tmp = i.min(tmp);
            }
            std::sync::Arc::get_mut(&mut out.data).unwrap()[0] = tmp;
        } else {
            let num_elems_reduced = <Src as HasAxes<Ax>>::size(&inp.shape);
            let inp_buf = inp.data.as_ref();
            let mut idx = index_for_reductions::<Src, Ax>(inp.shape, inp.strides);
            for o in out.buf_iter_mut() {
                let mut tmp: E = E::infinity();
                for _ in 0..num_elems_reduced {
                    tmp = tmp.min(inp_buf[idx.next().unwrap()]);
                }
                *o = tmp;
            }
        }
        Ok(out)
    }

    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        inp: &Self::Storage<Src, E>,
        grad_inp: &mut Self::Storage<Src, E>,
        out: &Self::Storage<Dst, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let num_elems_reduced = <Src as HasAxes<Ax>>::size(&grad_inp.shape);

        let grad_inp_buf = std::sync::Arc::make_mut(&mut grad_inp.data);
        let inp_buf = inp.data.as_ref();
        let mut inp_idx = index_for_reductions::<Src, Ax>(grad_inp.shape, grad_inp.strides);

        for (&o, &go) in out.buf_iter().zip(grad_out.buf_iter()) {
            for _ in 0..num_elems_reduced {
                let inp_i = inp_idx.next().unwrap();
                let d = if o == inp_buf[inp_i] {
                    E::one()
                } else {
                    E::zero()
                };
                grad_inp_buf[inp_i] += go * d;
            }
        }
        Ok(())
    }
}
