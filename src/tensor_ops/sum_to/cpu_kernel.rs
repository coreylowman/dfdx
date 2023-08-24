use crate::{
    dtypes::{Dtype, NotMixedPrecision},
    shapes::{Axes, HasAxes, ReduceShapeTo, Shape},
    tensor::{Cpu, Tensor, Tensorlike, ZerosTensor},
    tensor_ops::utilities::reduction_utils::index_for_reductions,
};

#[cfg(feature = "f16")]
impl super::SumKernel<crate::dtypes::AMP<crate::dtypes::f16>> for Cpu {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Tensor<Src, crate::dtypes::AMP<crate::dtypes::f16>, Self>,
    ) -> Result<Tensor<Dst, crate::dtypes::AMP<crate::dtypes::f16>, Self>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let mut out = self.try_zeros_like(&dst)?;
        if Dst::NUM_DIMS == 0 {
            debug_assert_eq!(out.data.len(), 1);

            let mut tmp = 0.0f32;
            for v in inp.buf_iter() {
                tmp += v.0.to_f32();
            }
            let scale = (inp.shape.num_elements() / inp.data.len()) as f32;
            std::sync::Arc::get_mut(&mut out.data).unwrap()[0] =
                crate::dtypes::AMP(crate::dtypes::f16::from_f32(tmp * scale));
        } else {
            let num_elems_reduced = <Src as HasAxes<Ax>>::size(&inp.shape);
            let inp_buf = inp.data.as_ref();
            let mut idx = index_for_reductions::<Src, Ax>(inp.shape, inp.strides);
            for o in out.buf_iter_mut() {
                let mut tmp = 0.0f32;
                for _ in 0..num_elems_reduced {
                    tmp += inp_buf[idx.next().unwrap()].0.to_f32();
                }
                *o = crate::dtypes::AMP(crate::dtypes::f16::from_f32(tmp));
            }
        }
        Ok(out)
    }
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        _dst: Dst,
        inp: &impl Tensorlike<Src, crate::dtypes::AMP<crate::dtypes::f16>, Self>,
        grad_inp: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        if Dst::NUM_DIMS == 0 {
            debug_assert_eq!(grad_out.len(), 1);
            let v = grad_out[0].0.to_f32();
            let scale = (inp.shape().num_elements() / inp.len()) as f32;
            for i in grad_inp.iter_mut() {
                i.0 += crate::dtypes::f16::from_f32(v * scale);
            }
        } else {
            let num_elems_reduced = <Src as HasAxes<Ax>>::size(inp.shape());
            let mut idx = index_for_reductions::<Src, Ax>(*inp.shape(), inp.strides());
            for &o in grad_out.iter() {
                for _ in 0..num_elems_reduced {
                    grad_inp[idx.next().unwrap()] += o;
                }
            }
        }
        Ok(())
    }
}

impl<E: Dtype + NotMixedPrecision> super::SumKernel<E> for Cpu {
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
            let scale = E::from_usize(inp.shape.num_elements() / inp.data.len()).unwrap();
            let mut tmp: E = Default::default();
            for v in inp.buf_iter() {
                tmp += *v;
            }
            std::sync::Arc::get_mut(&mut out.data).unwrap()[0] = tmp * scale;
        } else {
            let num_elems_reduced = <Src as HasAxes<Ax>>::size(&inp.shape);
            let inp_buf = inp.data.as_ref();
            let mut idx = index_for_reductions::<Src, Ax>(inp.shape, inp.strides);
            for o in out.buf_iter_mut() {
                let mut tmp: E = Default::default();
                for _ in 0..num_elems_reduced {
                    tmp += inp_buf[idx.next().unwrap()];
                }
                *o = tmp;
            }
        }
        Ok(out)
    }
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        _dst: Dst,
        inp: &impl Tensorlike<Src, E, Self>,
        grad_inp: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        if Dst::NUM_DIMS == 0 {
            debug_assert_eq!(grad_out.len(), 1);
            let v = grad_out[0];
            let scale = E::from_usize(inp.shape().num_elements() / inp.len()).unwrap();
            for i in grad_inp.iter_mut() {
                *i += v * scale;
            }
        } else {
            let num_elems_reduced = <Src as HasAxes<Ax>>::size(inp.shape());
            let mut idx = index_for_reductions::<Src, Ax>(*inp.shape(), inp.strides());
            for &o in grad_out.iter() {
                for _ in 0..num_elems_reduced {
                    grad_inp[idx.next().unwrap()] += o;
                }
            }
        }
        Ok(())
    }
}
