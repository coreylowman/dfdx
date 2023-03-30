use crate::{
    prelude::TensorFromVec,
    shapes::{Dtype, Shape},
    tensor::{cpu::LendingIterator, Quantize, QuantizedCpu, Tensor},
};

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> super::ConcatKernel<K::Value>
    for QuantizedCpu<K>
where
    K::Value: Dtype,
{
    fn forward<A: Shape, B: Shape>(
        &self,
        a: &Tensor<A, K::Value, Self>,
        b: &Tensor<B, K::Value, Self>,
    ) -> Result<Tensor<A::Catted, K::Value, Self>, Self::Err>
    where
        A: super::ConcatShape<B>,
    {
        let shape = a.shape.concat_shape(&b.shape);
        let mut data = std::vec::Vec::with_capacity(shape.num_elements());
        if a.strides == a.shape.strides() {
            data.extend(a.data.iter());
        } else {
            data.extend(a.as_vec());
        }
        if b.strides == b.shape.strides() {
            data.extend(b.data.iter());
        } else {
            data.extend(b.as_vec());
        }
        Ok(self.tensor_from_vec(data, shape))
    }
    fn backward<A: Shape, B: Shape>(
        &self,
        _: &Tensor<A, K::Value, Self>,
        grad_a: &mut Self::Storage,
        _: &Tensor<B, K::Value, Self>,
        grad_b: &mut Self::Storage,
        _: &Tensor<A::Catted, K::Value, Self>,
        grad_out: &Self::Storage,
    ) -> Result<(), Self::Err>
    where
        A: super::ConcatShape<B>,
    {
        let mut offset = 0;
        let mut iter_a = grad_a.iter_blocks_mut();
        while let Some(mut block) = iter_a.next() {
            for ga in block.iter_mut() {
                *ga += grad_out.get(offset).unwrap();
                offset += 1;
            }
        }
        let mut iter_b = grad_b.iter_blocks_mut();
        while let Some(mut block) = iter_b.next() {
            for gb in block.iter_mut() {
                *gb += grad_out.get(offset).unwrap();
                offset += 1;
            }
        }
        Ok(())
    }
}
