use crate::shapes::{Shape, Unit};
use crate::tensor::{cpu::LendingIterator, storage_traits::*, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    collections::BTreeMap,
    sync::{Arc, RwLock},
    vec::Vec,
};

#[cfg(feature = "no-std")]
use spin::Mutex;

#[cfg(not(feature = "no-std"))]
use std::sync::Mutex;

#[derive(Copy, Clone, Debug)]
pub(crate) struct BytesPtr(pub(crate) *mut u8);
unsafe impl Send for BytesPtr {}
unsafe impl Sync for BytesPtr {}

/// A device that stores data on the heap.
///
/// The [Default] impl seeds the underlying rng with seed of 0.
///
/// Use [Cpu::seed_from_u64] to control what seed is used.
#[derive(Clone, Debug)]
pub struct Cpu {
    pub(crate) rng: Arc<Mutex<StdRng>>,
    pub(crate) cache: Arc<RwLock<BTreeMap<usize, Vec<BytesPtr>>>>,
}

impl Default for Cpu {
    fn default() -> Self {
        Self {
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(0))),
            cache: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }
}

impl Cpu {
    /// Constructs rng with the given seed.
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
            cache: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CpuError {
    /// Device is out of memory
    OutOfMemory,
    /// Not enough elements were provided when creating a tensor
    WrongNumElements,
}

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfMemory => f.write_str("CpuError::OutOfMemory"),
            Self::WrongNumElements => f.write_str("CpuError::WrongNumElements"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CpuError {}

impl HasErr for Cpu {
    type Err = CpuError;
}

#[derive(Debug)]
pub struct CachableVec<E> {
    pub(crate) data: Vec<E>,
    pub(crate) destination: Arc<RwLock<BTreeMap<usize, Vec<BytesPtr>>>>,
}

impl<E: Clone> Clone for CachableVec<E> {
    fn clone(&self) -> Self {
        let numel = self.data.len();
        let num_bytes = std::mem::size_of::<E>() * numel;
        let reuse = {
            let cache = self.destination.read().unwrap();
            cache.contains_key(&num_bytes)
        };
        let data = if reuse {
            let mut cache = self.destination.write().unwrap();
            let items = cache.get_mut(&num_bytes).unwrap();
            let allocation: *mut u8 = items.pop().unwrap().0;
            if items.is_empty() {
                cache.remove(&num_bytes);
            }
            let mut data = unsafe { Vec::from_raw_parts(allocation as *mut E, numel, numel) };
            data.clone_from(&self.data);
            data
        } else {
            self.data.clone()
        };
        Self {
            data,
            destination: self.destination.clone(),
        }
    }
}

impl<E> Drop for CachableVec<E> {
    fn drop(&mut self) {
        let mut data = std::mem::take(&mut self.data);
        let num_bytes = data.len() * std::mem::size_of::<E>();

        data.shrink_to_fit();
        let ptr = data.as_mut_ptr() as *mut u8;
        std::mem::forget(data);

        let mut cache = self.destination.write().unwrap();
        if let std::collections::btree_map::Entry::Vacant(e) = cache.entry(num_bytes) {
            e.insert(std::vec![BytesPtr(ptr)]);
        } else {
            cache.get_mut(&num_bytes).unwrap().push(BytesPtr(ptr));
        }
    }
}

impl<E> std::ops::Deref for CachableVec<E> {
    type Target = Vec<E>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<E> std::ops::DerefMut for CachableVec<E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl DeviceStorage for Cpu {
    type Vec<E: Unit> = CachableVec<E>;

    fn try_alloc_len<E: Unit>(&self, len: usize) -> Result<Self::Vec<E>, Self::Err> {
        self.try_alloc_zeros(len)
    }

    fn random_u64(&self) -> u64 {
        #[cfg(not(feature = "no-std"))]
        {
            self.rng.lock().unwrap().gen()
        }
        #[cfg(feature = "no-std")]
        {
            self.rng.lock().gen()
        }
    }

    fn len<E: Unit>(&self, v: &Self::Vec<E>) -> usize {
        v.len()
    }

    fn tensor_to_vec<S: Shape, E: Unit, T>(&self, tensor: &Tensor<S, E, Self, T>) -> Vec<E> {
        let mut buf = Vec::with_capacity(tensor.shape.num_elements());
        let mut iter = tensor.iter();
        while let Some(v) = iter.next() {
            buf.push(*v);
        }
        buf
    }

    fn try_synchronize(&self) -> Result<(), Self::Err> {
        Ok(())
    }
}
