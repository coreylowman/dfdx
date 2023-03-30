use crate::{
    shapes::{Axes, Dtype, HasAxes, ReduceShapeTo, Shape},
    tensor::{cpu::LendingIterator, Quantize, QuantizedCpu, Tensor, ZerosTensor},
    tensor_ops::utilities::reduction_utils::index_for_reductions,
};

use num_traits::Float;

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> super::MaxReduceKernel<K::Value>
    for QuantizedCpu<K>
where
    K::Value: Dtype,
{
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Tensor<Src, K::Value, Self>,
    ) -> Result<Tensor<Dst, K::Value, Self>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let mut out = self.try_zeros_like(&dst)?;
        if Dst::NUM_DIMS == 0 {
            debug_assert_eq!(out.data.len(), 1);
            let mut tmp = K::Value::neg_infinity();
            for i in inp.buf_iter() {
                tmp = i.max(tmp);
            }
            *std::sync::Arc::get_mut(&mut out.data)
                .unwrap()
                .get_block_mut(0)
                .unwrap()
                .get_mut(0)
                .unwrap() = tmp;
        } else {
            let num_elems_reduced = <Src as HasAxes<Ax>>::size(&inp.shape);
            let inp_buf = inp.data.as_ref();
            let mut idx = index_for_reductions::<Src, Ax>(inp.shape, inp.strides);
            let mut blocks_iter = out.iter_blocks_mut();
            while let Some(mut block) = blocks_iter.next() {
                for o in block.iter_mut() {
                    let mut tmp = K::Value::neg_infinity();
                    for _ in 0..num_elems_reduced {
                        tmp = tmp.max(inp_buf.get(idx.next().unwrap()).unwrap());
                    }
                    *o = tmp;
                }
            }
        }
        Ok(out)
    }

    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        _inp: &Tensor<Src, K::Value, Self>,
        _grad_inp: &mut Self::Storage,
        _out: &Tensor<Dst, K::Value, Self>,
        _grad_out: &Self::Storage,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        // let num_elems_reduced = <Src as HasAxes<Ax>>::size(&inp.shape);

        // let inp_buf = inp.data.as_ref();
        // let mut inp_idx = index_for_reductions::<Src, Ax>(inp.shape, inp.strides);

        // for (o, go) in out.buf_iter().zip(grad_out.iter()) {
        //     for _ in 0..num_elems_reduced {
        //         let inp_i = inp_idx.next().unwrap();
        //         let d = if o == inp_buf.get(inp_i).unwrap() {
        //             <K::Value as num_traits::One>::one()
        //         } else {
        //             <K::Value as num_traits::Zero>::zero()
        //         };
        //         grad_inp.get_mut(inp_i).unwrap() += go * d;
        //     }
        // }
        // Ok(())
        unimplemented!();
    }
}
