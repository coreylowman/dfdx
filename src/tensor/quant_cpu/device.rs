use crate::{prelude::cpu::LendingIterator, shapes::Unit, tensor::storage_traits::*};
use core::{marker::PhantomData, simd::Simd};
use std::vec::Vec;

use super::{
    iterate::{
        QuantBlockMutRef, QuantBlockRef, QuantizedStorageIter, QuantizedStorageMutIter,
        QuantizedStorageRefIter,
    },
    quantize::{HalfBytePair, HalfByteQuantizer},
};

#[derive(Copy, Clone, Debug, Default)]
pub struct QuantStorage<K>(PhantomData<K>);

impl<K: HalfByteQuantizer + std::fmt::Debug + Send + Sync> DeviceStorage<K::Value>
    for QuantStorage<K>
{
    type Storage = QuantizedStorage<K>;
}

// TODO: It would be nice to be able to configure this, but const generics make it quite difficult.
const BLOCK_SIZE: usize = 32;

#[derive(Copy, Clone, Debug)]
pub struct HalfByteBlock<K: HalfByteQuantizer> {
    quants: Simd<u8, { BLOCK_SIZE / 2 }>,
    length: usize,
    kind: K,
}

impl<K: HalfByteQuantizer> HalfByteBlock<K> {
    pub fn from_block(values: &[K::Value]) -> Self {
        let length = values.len();
        let kind = K::from_values(values);
        let mut quants = [0u8; BLOCK_SIZE / 2];
        for (i, values) in values.chunks(2).enumerate() {
            let (v1, v2) = (values[0], values.get(1).copied().unwrap_or_default());
            quants[i] = u8::half_byte_pair(kind.quantize(v1), kind.quantize(v2));
        }
        Self {
            quants: Simd::from_array(quants),
            length,
            kind,
        }
    }

    pub fn get_values(&self) -> Vec<K::Value> {
        self.quants
            .as_array()
            .iter()
            .copied()
            .take(self.length)
            .flat_map(|quant| [quant.first(), quant.second()])
            .map(|half_byte| self.kind.dequantize(half_byte))
            .collect()
    }

    fn get(&self, index: usize) -> Option<K::Value> {
        if index < BLOCK_SIZE {
            let half_byte = if index % 2 == 0 {
                self.quants[index / 2].first()
            } else {
                self.quants[index / 2].second()
            };
            Some(self.kind.dequantize(half_byte))
        } else {
            None
        }
    }

    pub fn as_ref(&self) -> QuantBlockRef<'_, K> {
        QuantBlockRef::new(self)
    }

    pub fn as_mut(&mut self) -> QuantBlockMutRef<'_, K> {
        QuantBlockMutRef::new(self)
    }

    #[cfg(test)]
    fn size() -> usize {
        (BLOCK_SIZE / 2) + std::mem::size_of::<usize>() + std::mem::size_of::<K>()
    }
}

#[derive(Clone, Debug, Default)]
pub struct QuantizedStorage<K: HalfByteQuantizer> {
    pub(super) blocks: Vec<HalfByteBlock<K>>,
}

impl<K: HalfByteQuantizer> QuantizedStorage<K> {
    pub fn try_with_capacity(cap: usize) -> Result<Self, std::collections::TryReserveError> {
        let mut blocks = Vec::new();
        let mut capacity = cap / BLOCK_SIZE;
        if cap % BLOCK_SIZE != 0 {
            capacity += 1;
        }
        blocks.try_reserve(capacity)?;
        Ok(Self { blocks })
    }

    pub fn resize(&mut self, new_len: usize, value: K::Value) {
        if new_len / BLOCK_SIZE > 0 {
            let full_block = HalfByteBlock::from_block(&[value; BLOCK_SIZE]);
            self.blocks.resize(new_len / BLOCK_SIZE, full_block);
        }
        if new_len % BLOCK_SIZE > 0 {
            let partial_block = HalfByteBlock::from_block(&vec![value; new_len % BLOCK_SIZE]);
            self.blocks.resize(new_len / BLOCK_SIZE + 1, partial_block);
        }
    }

    pub fn from_slice(slice: &[K::Value]) -> Self {
        let mut res = Self::try_with_capacity(slice.len()).unwrap();
        for block in slice.chunks(BLOCK_SIZE) {
            res.blocks.push(HalfByteBlock::from_block(block));
        }
        res
    }

    pub fn copy_from_slice(&mut self, slice: &[K::Value]) {
        assert!(slice.len() == self.len());
        let mut slice_iter = slice.iter();
        let mut storage_iter = self.iter_mut();
        while let Some(val) = storage_iter.next() {
            *val = *Iterator::next(&mut slice_iter).unwrap();
        }
    }

    pub fn from_iter(iter: impl Iterator<Item = K::Value>, count: usize) -> Self {
        let mut res = Self::try_with_capacity(count).unwrap();
        let mut iter = iter.take(count).peekable();
        while iter.peek().is_some() {
            let block = iter.by_ref().take(BLOCK_SIZE).collect::<Vec<_>>();
            res.blocks.push(HalfByteBlock::from_block(&block));
        }
        res
    }

    pub fn fill(&mut self, value: K::Value) {
        let length = self.len();
        if length / BLOCK_SIZE > 0 {
            let full_block = HalfByteBlock::from_block(&[value; BLOCK_SIZE]);
            for i in 0..length / BLOCK_SIZE {
                self.blocks[i] = full_block.clone();
            }
        }
        if length % BLOCK_SIZE > 0 {
            let partial_block = HalfByteBlock::from_block(&vec![value; length % BLOCK_SIZE]);
            self.blocks[length / BLOCK_SIZE] = partial_block;
        }
    }

    pub fn get(&self, index: usize) -> Option<K::Value> {
        assert!(index < self.len());
        self.blocks[index / BLOCK_SIZE].get(index % BLOCK_SIZE)
    }

    pub fn iter(&self) -> QuantizedStorageRefIter<'_, K> {
        QuantizedStorageRefIter::new(self)
    }

    pub fn iter_mut(&mut self) -> QuantizedStorageMutIter<'_, K> {
        QuantizedStorageMutIter::new(self)
    }

    pub fn len(&self) -> usize {
        if !self.blocks.is_empty() {
            (self.blocks.len() - 1) * BLOCK_SIZE + self.blocks.last().unwrap().length
        } else {
            0
        }
    }

    #[cfg(test)]
    pub fn size(&self) -> usize {
        self.blocks.capacity() * HalfByteBlock::<K>::size()
            + std::mem::size_of::<Vec<HalfByteBlock<K>>>()
    }
}

impl<K: HalfByteQuantizer> IntoIterator for QuantizedStorage<K> {
    type Item = K::Value;
    type IntoIter = QuantizedStorageIter<K>;

    fn into_iter(self) -> Self::IntoIter {
        QuantizedStorageIter::new(self)
    }
}

impl<E: Unit, K: HalfByteQuantizer<Value = E> + std::fmt::Debug + Send + Sync> Storage<E>
    for QuantizedStorage<K>
{
    type Iter<'a> = QuantizedStorageRefIter<'a, K>
    where
        Self: 'a;

    type IterMut<'a> = QuantizedStorageMutIter<'a, K>
    where
        Self: 'a;
    type Err = std::collections::TryReserveError;

    fn try_alloc_elem(numel: usize, elem: E) -> Result<Self, Self::Err> {
        #[cfg(feature = "fast-alloc")]
        {
            Ok(QuantizedStorage::from_iter(
                core::iter::repeat_with(|| elem),
                numel,
            ))
        }

        #[cfg(not(feature = "fast-alloc"))]
        {
            let mut data =
                QuantizedStorage::try_with_capacity(numel).map_err(|_| CpuError::OutOfMemory)?;
            data.resize(numel, elem);
            Ok(data)
        }
    }

    fn fill(&mut self, value: E) {
        self.fill(value)
    }

    fn from_vec(vec: Vec<E>) -> Self {
        Self::from_slice(&vec)
    }

    fn index(&self, index: usize) -> E {
        self.get(index).unwrap()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.iter_mut()
    }
}
