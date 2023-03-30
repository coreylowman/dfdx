use crate::tensor::{cpu::LendingIterator, Quantize, QuantizedCpu};

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> super::AxpyKernel<K::Value>
    for QuantizedCpu<K>
{
    fn forward(
        &self,
        a: &mut Self::Storage,
        alpha: K::Value,
        b: &Self::Storage,
        beta: K::Value,
    ) -> Result<(), Self::Err> {
        let mut iter = a.iter_blocks_mut();
        while let Some(mut block) = iter.next() {
            for (a_i, b_i) in block.iter_mut().zip(b.iter()) {
                *a_i = *a_i * alpha + b_i * beta;
            }
        }
        Ok(())
    }
}
