use crate::{shapes::Shape, tensor::Tensor};

use super::{Quantize, QuantizedCpu};

pub(crate) fn index_to_i<S: Shape>(shape: &S, strides: &S::Concrete, index: S::Concrete) -> usize {
    let sizes = shape.concrete();
    for (i, idx) in index.into_iter().enumerate() {
        if idx >= sizes[i] {
            panic!("Index out of bounds: index={index:?} shape={shape:?}");
        }
    }
    strides
        .into_iter()
        .zip(index.into_iter())
        .map(|(a, b)| a * b)
        .sum()
}

impl<S: Shape, K: 'static + Quantize + std::fmt::Debug + Send + Sync, T>
    Tensor<S, K::Value, QuantizedCpu<K>, T>
{
    #[inline(always)]
    pub fn index(&self, index: S::Concrete) -> K::Value {
        let i = index_to_i(&self.shape, &self.strides, index);
        self.data.get(i).unwrap()
    }
}
