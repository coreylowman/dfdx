use crate::{
    shapes::{Axes, Dtype, HasAxes, ReduceShapeTo, Shape},
    tensor::cpu::{Cpu, LendingIterator, StridedArray},
    tensor_ops::utilities::reduction_utils::index_for_reductions,
};

impl<E: Dtype> super::SumKernel<E> for Cpu {
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
            let mut tmp: E = Default::default();
            let mut inp_iter = inp.iter();
            while let Some(i) = inp_iter.next() {
                tmp += *i;
            }
            std::sync::Arc::get_mut(&mut out.data).unwrap()[0] = tmp;
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
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        if Dst::NUM_DIMS == 0 {
            debug_assert_eq!(grad_out.data.len(), 1);
            let v = grad_out.data[0];
            let scale = E::from_usize(grad_inp.shape.num_elements() / grad_inp.data.len()).unwrap();
            for i in grad_inp.buf_iter_mut() {
                *i += v * scale;
            }
        } else {
            let num_elems_reduced = <Src as HasAxes<Ax>>::size(&grad_inp.shape);
            let inp_buf = std::sync::Arc::make_mut(&mut grad_inp.data);
            let mut idx = index_for_reductions::<Src, Ax>(grad_inp.shape, grad_inp.strides);
            for &o in grad_out.buf_iter() {
                for _ in 0..num_elems_reduced {
                    inp_buf[idx.next().unwrap()] += o;
                }
            }
        }
        Ok(())
    }
}
