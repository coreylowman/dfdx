use crate::tensor::cpu::LendingIterator;
use crate::{
    shapes::{Axes, Dtype, HasAxes, ReduceShapeTo, Shape},
    tensor::{Quantize, QuantizedCpu, Tensor, ZerosTensor},
    tensor_ops::utilities::reduction_utils::index_for_reductions,
};

use num_traits::FromPrimitive;

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> super::SumKernel<K::Value>
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
            let scale = K::Value::from_usize(inp.shape.num_elements() / inp.data.len()).unwrap();
            let mut tmp: K::Value = Default::default();
            for v in inp.buf_iter() {
                tmp += v;
            }
            *std::sync::Arc::get_mut(&mut out.data)
                .unwrap()
                .get_block_mut(0)
                .unwrap()
                .get_mut(0)
                .unwrap() = tmp * scale;
        } else {
            let num_elems_reduced = <Src as HasAxes<Ax>>::size(&inp.shape);
            let mut idx = index_for_reductions::<Src, Ax>(inp.shape, inp.strides);
            let mut blocks_iter = out.iter_blocks_mut();
            while let Some(mut block) = blocks_iter.next() {
                for o in block.iter_mut() {
                    let mut tmp: K::Value = Default::default();
                    for _ in 0..num_elems_reduced {
                        tmp += inp.data.get(idx.next().unwrap()).unwrap();
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
        _: &Tensor<Dst, K::Value, Self>,
        _grad_out: &Self::Storage,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        // if Dst::NUM_DIMS == 0 {
        //     debug_assert_eq!(grad_out.len(), 1);
        //     let v = grad_out[0];
        //     let scale = K::Value::from_usize(inp.shape.num_elements() / inp.data.len()).unwrap();
        //     for i in grad_inp.iter_mut() {
        //         *i += v * scale;
        //     }
        // } else {
        //     let num_elems_reduced = <Src as HasAxes<Ax>>::size(&inp.shape);
        //     let mut idx = index_for_reductions::<Src, Ax>(inp.shape, inp.strides);
        //     for &o in grad_out.iter() {
        //         for _ in 0..num_elems_reduced {
        //             grad_inp[idx.next().unwrap()] += o;
        //         }
        //     }
        // }
        // Ok(())
        unimplemented!();
    }
}
