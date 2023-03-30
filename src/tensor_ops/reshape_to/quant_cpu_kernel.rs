use crate::shapes::{Dtype, Shape};
use crate::tensor::{cpu::LendingIterator, Quantize, QuantizedCpu, Tensor, ZerosTensor};

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> super::ReshapeKernel<K::Value>
    for QuantizedCpu<K>
where
    K::Value: Dtype,
{
    fn forward<Src: Shape, Dst: Shape>(
        &self,
        dst: &Dst,
        inp: &Tensor<Src, K::Value, Self>,
    ) -> Result<Tensor<Dst, K::Value, Self>, Self::Err> {
        let mut out = self.try_zeros_like(dst)?;
        let mut out_iter = out.iter_blocks_mut();
        while let Some(mut block) = out_iter.next() {
            for (o, i) in block.iter_mut().zip(inp.iter()) {
                *o = i;
            }
        }
        Ok(out)
    }
    fn backward<Src: Shape, Dst: Shape>(
        &self,
        _inp: &Tensor<Src, K::Value, Self>,
        _grad_inp: &mut Self::Storage,
        _out: &Tensor<Dst, K::Value, Self>,
        _grad_out: &Self::Storage,
    ) -> Result<(), Self::Err> {
        unimplemented!();
    }
}
