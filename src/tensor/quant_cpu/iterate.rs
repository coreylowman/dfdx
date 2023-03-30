use super::{
    super::Tensor,
    device::{QuantizedStorage, QuantizedStorageBlocksIter, QuantizedStorageRefIter},
    Quantize, QuantizedCpu,
};
use crate::{prelude::cpu::NdIndex, shapes::Shape};

pub(crate) struct StridedRefIter<'a, S: Shape, K: Quantize> {
    data: &'a QuantizedStorage<K>,
    index: NdIndex<S>,
}

pub(crate) struct StridedRefIndexIter<'a, S: Shape, K: Quantize> {
    data: &'a QuantizedStorage<K>,
    index: NdIndex<S>,
}

impl<S: Shape, K: 'static + Quantize + std::fmt::Debug + Send + Sync, T>
    Tensor<S, K::Value, QuantizedCpu<K>, T>
{
    #[inline]
    pub(crate) fn buf_iter(&self) -> QuantizedStorageRefIter<K> {
        self.data.iter()
    }

    #[inline]
    pub(crate) fn iter_blocks_mut(&mut self) -> QuantizedStorageBlocksIter<K> {
        std::sync::Arc::make_mut(&mut self.data).iter_blocks_mut()
    }

    #[inline]
    pub(crate) fn iter(&self) -> StridedRefIter<S, K> {
        StridedRefIter {
            data: self.data.as_ref(),
            index: NdIndex::new(self.shape, self.strides),
        }
    }

    #[inline]
    pub(crate) fn iter_with_index(&self) -> StridedRefIndexIter<S, K> {
        StridedRefIndexIter {
            data: self.data.as_ref(),
            index: NdIndex::new(self.shape, self.strides),
        }
    }
}

impl<'q, S: Shape, K: Quantize> Iterator for StridedRefIter<'q, S, K> {
    type Item = K::Value;

    fn next(&mut self) -> Option<Self::Item> {
        self.index.next().and_then(|i| self.data.get(i))
    }
}

impl<'q, S: Shape, K: Quantize> Iterator for StridedRefIndexIter<'q, S, K> {
    type Item = (K::Value, S::Concrete);

    fn next(&mut self) -> Option<Self::Item> {
        self.index
            .next_with_idx()
            .and_then(|(i, idx)| self.data.get(i).map(|d| (d, idx)))
    }
}
