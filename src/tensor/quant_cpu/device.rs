use crate::prelude::cpu::LendingIterator;
use crate::prelude::{Cpu, CpuError};
use crate::shapes::Shape;
use crate::tensor::{storage_traits::*, Tensor};

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use std::simd::Simd;
use std::vec::Vec;

use super::quantize::{u4, Quantize};

/// A mutable reference to a block of quantized values. Values must be edited as a block because
/// value changes imply changes to the calculated quantization factors. The new block is computed
/// when this value is dropped, and stored in the underlying memory.
pub struct QuantBlockMutRef<'q, K: Quantize> {
    inner: &'q mut QuantBlock<K>,
    vals: Vec<K::Value>,
}

impl<'q, K: Quantize> Deref for QuantBlockMutRef<'q, K> {
    type Target = Vec<K::Value>;

    fn deref(&self) -> &Self::Target {
        &self.vals
    }
}

impl<'q, K: Quantize> DerefMut for QuantBlockMutRef<'q, K> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vals
    }
}

impl<'q, K: Quantize> Drop for QuantBlockMutRef<'q, K> {
    fn drop(&mut self) {
        *self.inner = QuantBlock::from_block(&self.vals);
    }
}

/// A pair of half-byte values stored as one byte.
/// Utilizes a trait to make usage simpler.
trait HalfBytePair {
    fn half_byte_pair(first: u4, second: u4) -> Self;

    /// Gets the first half-byte.
    fn first(&self) -> u4;

    /// Gets the second half-byte.
    fn second(&self) -> u4;

    /// Sets the first half-byte, leaving the rest alone.
    fn set_first(&mut self, value: u4);

    /// Sets the second half-byte, leaving the rest alone.
    fn set_second(&mut self, value: u4);
}
impl HalfBytePair for u8 {
    fn half_byte_pair(first: u4, second: u4) -> Self {
        (first.0 << 4) & second.0
    }

    fn first(&self) -> u4 {
        u4(self >> 4)
    }

    fn second(&self) -> u4 {
        u4(self & 0b_0000_1111)
    }

    fn set_first(&mut self, value: u4) {
        *self &= 0b_0000_1111;
        *self |= value.0 << 4
    }

    fn set_second(&mut self, value: u4) {
        *self &= 0b_1111_0000;
        *self |= value.0
    }
}

// TODO: It would be nice to be able to configure this, but const generics make it quite difficult.
const BLOCK_SIZE: usize = 32;

#[derive(Copy, Clone, Debug)]
pub struct QuantBlock<K> {
    quants: Simd<u8, { BLOCK_SIZE / 2 }>,
    length: usize,
    kind: K,
}

impl<K: Quantize> QuantBlock<K> {
    fn from_block(values: &[K::Value]) -> Self {
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

    fn get_values(&self) -> Vec<K::Value> {
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

    fn as_mut(&mut self) -> QuantBlockMutRef<'_, K> {
        QuantBlockMutRef {
            vals: self.get_values(),
            inner: self,
        }
    }

    #[cfg(test)]
    fn size() -> usize {
        (BLOCK_SIZE / 2) + std::mem::size_of::<usize>() + std::mem::size_of::<K>()
    }
}

#[derive(Clone, Debug, Default)]
pub struct QuantizedStorage<K> {
    pub(super) blocks: Vec<QuantBlock<K>>,
}

impl<K: Quantize> QuantizedStorage<K> {
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
            let full_block = QuantBlock::from_block(&[value; BLOCK_SIZE]);
            self.blocks.resize(new_len / BLOCK_SIZE, full_block);
        }
        if new_len % BLOCK_SIZE > 0 {
            let partial_block = QuantBlock::from_block(&vec![value; new_len % BLOCK_SIZE]);
            self.blocks.resize(new_len / BLOCK_SIZE + 1, partial_block);
        }
    }

    pub fn from_slice(slice: &[K::Value]) -> Self {
        let mut res = Self::try_with_capacity(slice.len()).unwrap();
        for block in slice.chunks(BLOCK_SIZE) {
            res.blocks.push(QuantBlock::from_block(block));
        }
        res
    }

    pub fn from_iter(iter: impl Iterator<Item = K::Value>, count: usize) -> Self {
        let mut res = Self::try_with_capacity(count).unwrap();
        let mut iter = iter.take(count).peekable();
        while iter.peek().is_some() {
            let block = iter.by_ref().take(BLOCK_SIZE).collect::<Vec<_>>();
            res.blocks.push(QuantBlock::from_block(&block));
        }
        res
    }

    pub fn fill(&mut self, value: K::Value) {
        let length = self.len();
        if length / BLOCK_SIZE > 0 {
            let full_block = QuantBlock::from_block(&[value; BLOCK_SIZE]);
            for i in 0..length / BLOCK_SIZE {
                self.blocks[i] = full_block.clone();
            }
        }
        if length % BLOCK_SIZE > 0 {
            let partial_block = QuantBlock::from_block(&vec![value; length % BLOCK_SIZE]);
            self.blocks[length / BLOCK_SIZE] = partial_block;
        }
    }

    pub fn get(&self, index: usize) -> Option<K::Value> {
        assert!(index < self.len());
        self.blocks[index / BLOCK_SIZE].get(index % BLOCK_SIZE)
    }

    pub fn get_block_mut(&mut self, index: usize) -> Option<QuantBlockMutRef<'_, K>> {
        self.blocks
            .get_mut(index / BLOCK_SIZE)
            .map(|block| block.as_mut())
    }

    pub fn iter(&self) -> QuantizedStorageRefIter<'_, K> {
        QuantizedStorageRefIter {
            inner: self,
            pos: 0,
        }
    }

    pub fn iter_blocks_mut(&mut self) -> QuantizedStorageBlocksIter<'_, K> {
        QuantizedStorageBlocksIter {
            inner: self,
            pos: 0,
        }
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
        self.blocks.capacity() * QuantBlock::<K>::size() + std::mem::size_of::<Vec<QuantBlock<K>>>()
    }
}

impl<K: Quantize> IntoIterator for QuantizedStorage<K> {
    type Item = K::Value;
    type IntoIter = QuantizedStorageIter<K>;

    fn into_iter(self) -> Self::IntoIter {
        QuantizedStorageIter {
            inner: self,
            pos: 0,
        }
    }
}

#[derive(Clone)]
pub struct QuantizedStorageIter<K: Quantize> {
    inner: QuantizedStorage<K>,
    pos: usize,
}

impl<K: Quantize> Iterator for QuantizedStorageIter<K> {
    type Item = K::Value;

    fn next(&mut self) -> Option<Self::Item> {
        let res = self.inner.get(self.pos);
        self.pos += 1;
        res
    }
}

#[derive(Clone)]
pub struct QuantizedStorageRefIter<'q, K: Quantize> {
    inner: &'q QuantizedStorage<K>,
    pos: usize,
}

impl<'q, K: Quantize> Iterator for QuantizedStorageRefIter<'q, K> {
    type Item = K::Value;

    fn next(&mut self) -> Option<Self::Item> {
        let res = self.inner.get(self.pos);
        self.pos += 1;
        res
    }
}

pub struct QuantizedStorageBlocksIter<'q, K: Quantize> {
    inner: &'q mut QuantizedStorage<K>,
    pos: usize,
}

impl<'q, K: Quantize> LendingIterator for QuantizedStorageBlocksIter<'q, K> {
    type Item<'a> = QuantBlockMutRef<'a, K>
    where
        Self: 'a;

    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        if let Some(block) = self.inner.blocks.get_mut(self.pos) {
            self.pos += 1;
            Some(block.as_mut())
        } else {
            None
        }
    }
}

/// A device that stores data on the heap.
///
/// The [Default] impl seeds the underlying rng with seed of 0.
///
/// Use [`QuantizedCpu::seed_from_u64`] to control what seed is used.
pub struct QuantizedCpu<K> {
    pub cpu: Cpu,
    _quant_kind: PhantomData<K>,
}

impl<K> QuantizedCpu<K> {
    /// Constructs rng with the given seed.
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            cpu: Cpu::seed_from_u64(seed),
            ..Default::default()
        }
    }
}

impl<K> Clone for QuantizedCpu<K> {
    fn clone(&self) -> Self {
        Self {
            cpu: self.cpu.clone(),
            _quant_kind: PhantomData,
        }
    }
}

impl<K> core::fmt::Debug for QuantizedCpu<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QuantizedCpu")
            .field("cpu", &self.cpu)
            .finish()
    }
}

impl<K> Default for QuantizedCpu<K> {
    fn default() -> Self {
        Self {
            cpu: Default::default(),
            _quant_kind: Default::default(),
        }
    }
}

impl<K> HasErr for QuantizedCpu<K> {
    type Err = CpuError;
}

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> DeviceStorage<K::Value>
    for QuantizedCpu<K>
{
    type Storage = QuantizedStorage<K>;

    fn random_u64(&self) -> u64 {
        DeviceStorage::<K::Value>::random_u64(&self.cpu)
    }
}

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> DeviceAllocGrad<K::Value>
    for QuantizedCpu<K>
{
    fn try_alloc_grad(&self, other: &Self::Storage) -> Result<Self::Storage, Self::Err> {
        self.try_alloc_zeros(other.len())
    }
}

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> DeviceTensorToVec<K::Value>
    for QuantizedCpu<K>
{
    fn tensor_to_vec<S: Shape, T>(&self, tensor: &Tensor<S, K::Value, Self, T>) -> Vec<K::Value> {
        let mut buf = Vec::with_capacity(tensor.shape.num_elements());
        let mut iter = tensor.iter();
        while let Some(v) = iter.next() {
            buf.push(v);
        }
        buf
    }
}
