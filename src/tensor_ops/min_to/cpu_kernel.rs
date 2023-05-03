use crate::{
    shapes::{Axes, Dtype, HasAxes, ReduceShapeTo, Shape},
    tensor::{Cpu, Tensor, ZerosTensor},
    tensor_ops::utilities::reduction_utils::index_for_reductions,
};

use num_traits::Float;

impl<E: Dtype + Float> super::MinReduceKernel<E> for Cpu {
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
            let mut tmp: E = E::infinity();
            for i in inp.buf_iter() {
                tmp = i.min(tmp);
            }
            std::sync::Arc::get_mut(&mut out.data).unwrap()[0] = tmp;
        } else {
            let num_elems_reduced = <Src as HasAxes<Ax>>::size(&inp.shape);
            let inp_buf = inp.data.as_ref();
            #[cfg(not(feature = "threaded"))]
            {
                let mut idx = index_for_reductions::<Src, Ax>(inp.shape, inp.strides);
                for o in out.buf_iter_mut() {
                    let mut tmp: E = E::infinity();
                    for _ in 0..num_elems_reduced {
                        tmp = tmp.min(inp_buf[idx.next().unwrap()]);
                    }
                    *o = tmp;
                }
            }

            #[cfg(feature = "threaded")]
            {
                use rayon::prelude::*;
                let idx = index_for_reductions::<Src, Ax>(inp.shape, inp.strides);
                let buf = std::sync::Arc::make_mut(&mut out.data);
                buf.par_iter_mut().enumerate().for_each(|(i, o)| {
                    let mut tmp: E = E::infinity();
                    for j in 0..num_elems_reduced {
                        tmp = tmp.min(inp_buf[idx.get_strided_index(i * num_elems_reduced + j)]);
                    }
                    *o = tmp;
                });
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
        let inp_buf = inp.data.as_ref();

        #[cfg(not(feature = "threaded"))]
        {
            let num_elems_reduced = <Src as HasAxes<Ax>>::size(&inp.shape);
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
        }

        #[cfg(feature = "threaded")]
        {
            use crate::shapes::{BroadcastStridesTo, ReduceStridesTo};
            use rayon::prelude::*;
            let num_broadcasted = E::from_usize(inp.shape.num_elements() / grad_inp.len()).unwrap();

            let idx = index_for_reductions::<Src, Ax>(inp.shape, inp.strides);
            let dst: Dst = inp.shape.reduced();
            let out_strides = BroadcastStridesTo::<Src, Ax>::broadcast_strides(&dst, dst.strides());
            let out_idx = index_for_reductions::<Src, Ax>(inp.shape, out_strides);
            let out_strides = out_idx.strides;
            grad_inp.par_iter_mut().enumerate().for_each(|(i, gi)| {
                let out_i = idx.restride(i, out_strides);
                let go = grad_out[out_i];
                let o = out.data[out_i];
                let d = if o == inp_buf[i] { E::one() } else { E::zero() };
                *gi += go * d * num_broadcasted;
            });
        }
        Ok(())
    }
}
