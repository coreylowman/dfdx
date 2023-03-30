use crate::{
    prelude::TensorFromVec,
    shapes::*,
    tensor::{cpu::LendingIterator, Quantize, QuantizedCpu, Tensor},
};

use std::vec::Vec;

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> super::StackKernel<K::Value>
    for QuantizedCpu<K>
where
    K::Value: Dtype,
{
    fn forward<S: Shape, Num: Dim>(
        &self,
        num: Num,
        inp: &[Tensor<S, K::Value, Self>],
    ) -> Result<Tensor<S::Larger, K::Value, Self>, Self::Err>
    where
        S: super::AddDim<Num>,
    {
        debug_assert_eq!(inp.len(), num.size());

        // check that all the strides are the same
        let item_strides = inp[0].strides;
        for i in inp.iter() {
            assert_eq!(i.strides, item_strides);
        }
        let shape: S::Larger = inp[0].shape().add_dim(num);

        // build the new strides
        let mut strides = shape.strides();
        strides[0] = inp[0].data.len();
        for d in 1..<S::Larger as Shape>::NUM_DIMS {
            strides[d] = item_strides[d - 1];
        }

        // copy the data
        let numel = strides[0];
        let mut data: std::vec::Vec<K::Value> = std::vec::Vec::with_capacity(numel);
        for i in inp {
            data.extend(i.iter());
        }

        Ok(self.tensor_from_vec(data, shape))
    }
    fn backward(
        &self,
        mut grad_inp: Vec<&mut Self::Storage>,
        grad_out: &Self::Storage,
    ) -> Result<(), Self::Err> {
        let mut offset = 0;
        for item in grad_inp.drain(..) {
            let mut iter = item.iter_blocks_mut();
            while let Some(mut block) = iter.next() {
                for gi in block.iter_mut() {
                    *gi += grad_out.get(offset).unwrap();
                    offset += 1;
                }
            }
        }
        Ok(())
    }
}
