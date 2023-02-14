use crate::{
    shapes::*,
    tensor::cpu::{Cpu, StridedArray},
};

use std::vec::Vec;

impl<E: Dtype> super::StackKernel<E> for Cpu {
    fn forward<S: Shape, Num: Dim>(
        &self,
        num: Num,
        inp: Vec<&Self::Storage<S, E>>,
    ) -> Result<Self::Storage<S::Larger, E>, Self::Err>
    where
        S: super::AddDim<Num>,
    {
        debug_assert_eq!(inp.len(), num.size());

        // check that all the strides are the same
        let item_strides = inp[0].strides;
        for i in inp.iter() {
            assert_eq!(i.strides, item_strides);
        }
        let shape: S::Larger = inp[0].shape().add(num);

        // build the new strides
        let mut strides = shape.strides();
        strides[0] = inp[0].data.len();
        for d in 1..<S::Larger as Shape>::NUM_DIMS {
            strides[d] = item_strides[d - 1];
        }

        // copy the data
        let numel = strides[0];
        let mut data: std::vec::Vec<E> = std::vec::Vec::with_capacity(numel);
        for i in inp {
            data.extend_from_slice(i.data.as_ref());
        }

        Ok(StridedArray {
            data: std::sync::Arc::new(data),
            shape,
            strides,
        })
    }
    fn backward<S: Shape, New: Dim>(
        &self,
        mut grad_inp: Vec<&mut Self::Storage<S, E>>,
        grad_out: &Self::Storage<S::Larger, E>,
    ) -> Result<(), Self::Err>
    where
        S: super::AddDim<New>,
    {
        let grad_out_buf = grad_out.data.as_ref();
        let mut offset = 0;
        for item in grad_inp.drain(..) {
            for gi in item.buf_iter_mut() {
                *gi += grad_out_buf[offset];
                offset += 1;
            }
        }
        Ok(())
    }
}
