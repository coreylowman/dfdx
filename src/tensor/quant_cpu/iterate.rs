use core::ops::{Deref, DerefMut};

use crate::prelude::cpu::LendingIterator;

use super::{
    device::{HalfByteBlock, QuantizedStorage},
    HalfByteQuantizer,
};

/// A mutable reference to a block of quantized values. Values must be edited as a block because
/// value changes imply changes to the calculated quantization factors. The new block is computed
/// when this value is dropped, and stored in the underlying memory.
pub struct QuantBlockMutRef<'q, K: HalfByteQuantizer> {
    inner: &'q mut HalfByteBlock<K>,
    vals: Vec<K::Value>,
}

impl<'q, K: HalfByteQuantizer> QuantBlockMutRef<'q, K> {
    pub fn new(block: &'q mut HalfByteBlock<K>) -> Self {
        Self {
            vals: block.get_values(),
            inner: block,
        }
    }
}

impl<'q, K: HalfByteQuantizer> Deref for QuantBlockMutRef<'q, K> {
    type Target = Vec<K::Value>;

    fn deref(&self) -> &Self::Target {
        &self.vals
    }
}

impl<'q, K: HalfByteQuantizer> DerefMut for QuantBlockMutRef<'q, K> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vals
    }
}

impl<'q, K: HalfByteQuantizer> Drop for QuantBlockMutRef<'q, K> {
    fn drop(&mut self) {
        *self.inner = HalfByteBlock::from_block(&self.vals);
    }
}

#[derive(Clone)]
pub struct QuantBlockRef<'q, K: HalfByteQuantizer> {
    _inner: &'q HalfByteBlock<K>,
    vals: Vec<K::Value>,
}

impl<'q, K: HalfByteQuantizer> QuantBlockRef<'q, K> {
    pub fn new(block: &'q HalfByteBlock<K>) -> Self {
        Self {
            vals: block.get_values(),
            _inner: block,
        }
    }
}

impl<'q, K: HalfByteQuantizer> Deref for QuantBlockRef<'q, K> {
    type Target = Vec<K::Value>;

    fn deref(&self) -> &Self::Target {
        &self.vals
    }
}

#[derive(Clone)]
pub struct QuantizedStorageIter<K: HalfByteQuantizer> {
    inner: QuantizedStorage<K>,
    pos: usize,
}

impl<K: HalfByteQuantizer> QuantizedStorageIter<K> {
    pub fn new(storage: QuantizedStorage<K>) -> Self {
        Self {
            inner: storage,
            pos: 0,
        }
    }
}

impl<K: HalfByteQuantizer> Iterator for QuantizedStorageIter<K> {
    type Item = K::Value;

    fn next(&mut self) -> Option<Self::Item> {
        let res = self.inner.get(self.pos);
        self.pos += 1;
        res
    }
}

#[derive(Clone)]
pub struct QuantizedStorageRefIter<'q, K: HalfByteQuantizer> {
    blocks: Vec<QuantBlockRef<'q, K>>,
    block: QuantBlockRef<'q, K>,
    pos: usize,
}

impl<'q, K: HalfByteQuantizer> QuantizedStorageRefIter<'q, K> {
    pub fn new(storage: &'q QuantizedStorage<K>) -> Self {
        let mut blocks = storage
            .blocks
            .iter()
            .rev()
            .map(|b| b.as_ref())
            .collect::<Vec<_>>();
        Self {
            block: blocks.pop().unwrap(),
            blocks,
            pos: 0,
        }
    }
}

impl<'q, K: HalfByteQuantizer> LendingIterator for QuantizedStorageRefIter<'q, K> {
    type Item<'a> = &'a K::Value where Self: 'a;

    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        while self.block.get(self.pos).is_none() {
            if self.blocks.is_empty() {
                return None;
            } else {
                self.pos = 0;
                self.block = self.blocks.pop().unwrap();
            }
        }
        self.block.get(self.pos)
    }
}

pub struct QuantizedStorageMutIter<'q, K: HalfByteQuantizer> {
    blocks: Vec<QuantBlockMutRef<'q, K>>,
    block: QuantBlockMutRef<'q, K>,
    pos: usize,
}

impl<'q, K: HalfByteQuantizer> QuantizedStorageMutIter<'q, K> {
    pub fn new(storage: &'q mut QuantizedStorage<K>) -> Self {
        let mut blocks = storage
            .blocks
            .iter_mut()
            .rev()
            .map(|b| b.as_mut())
            .collect::<Vec<_>>();
        Self {
            block: blocks.pop().unwrap(),
            blocks,
            pos: 0,
        }
    }
}

impl<'q, K: HalfByteQuantizer> LendingIterator for QuantizedStorageMutIter<'q, K> {
    type Item<'a> = &'a mut K::Value where Self: 'a;

    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        while self.block.get(self.pos).is_none() {
            if self.blocks.is_empty() {
                return None;
            } else {
                self.pos = 0;
                self.block = self.blocks.pop().unwrap();
            }
        }
        self.block.get_mut(self.pos)
    }
}
