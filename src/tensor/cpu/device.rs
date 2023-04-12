use crate::shapes::{Shape, Unit};
use crate::tensor::{cache::TensorCache, cpu::LendingIterator, storage_traits::*, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{sync::Arc, vec::Vec};

#[cfg(feature = "no-std")]
use spin::Mutex;

#[cfg(not(feature = "no-std"))]
use std::sync::Mutex;

/// A pointer to a block of bytes on the heap. Used in conjunction with [TensorCache].
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
    /// A thread safe random number generator.
    pub(crate) rng: Arc<Mutex<StdRng>>,
    /// A thread safe cache of memory allocations that can be reused.
    pub(crate) cache: Arc<TensorCache<BytesPtr>>,
}

impl Default for Cpu {
    fn default() -> Self {
        Self {
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(0))),
            cache: Arc::new(Default::default()),
        }
    }
}

impl Cpu {
    /// Constructs rng with the given seed.
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
            cache: Arc::new(Default::default()),
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

/// A [Vec] that can be cloned without allocating new memory.
/// When [Drop]ed it will insert it's data into the cache.
#[derive(Debug)]
pub struct CachableVec<E> {
    /// The data stored in this vector.
    pub(crate) data: Vec<E>,
    /// A cache of memory allocations that can be reused.
    pub(crate) cache: Arc<TensorCache<BytesPtr>>,
}

impl<E: Clone> Clone for CachableVec<E> {
    fn clone(&self) -> Self {
        let numel = self.data.len();
        self.cache.try_pop::<E>(numel).map_or_else(
            || Self {
                data: self.data.clone(),
                cache: self.cache.clone(),
            },
            |allocation| {
                assert!(numel < isize::MAX as usize);
                // SAFETY:
                // - ✅ "ptr must have been allocated using the global allocator, such as via the alloc::alloc function."
                // - ✅ handled by tensor cache "T needs to have the same alignment as what ptr was allocated with."
                // - ✅ handled by tensor cache "The size of T times the capacity needs to be the same size as the pointer was allocated with."
                // - ✅ "length needs to be less than or equal to capacity."
                // - ✅ all the dtypes for this are builtin numbers "The first length values must be properly initialized values of type T."
                // - ✅ "capacity needs to be the capacity that the pointer was allocated with."
                // - ✅ "The allocated size in bytes must be no larger than isize::MAX. See the safety documentation of pointer::offset."
                let mut data = unsafe { Vec::from_raw_parts(allocation.0 as *mut E, numel, numel) };
                data.clone_from(&self.data);
                Self {
                    data,
                    cache: self.cache.clone(),
                }
            },
        )
    }
}

impl<E> Drop for CachableVec<E> {
    fn drop(&mut self) {
        if self.cache.is_enabled() {
            let mut data = std::mem::take(&mut self.data);
            data.shrink_to_fit();

            let numel = data.len();
            let ptr = data.as_mut_ptr() as *mut u8;
            std::mem::forget(data);

            self.cache.insert::<E>(numel, BytesPtr(ptr));
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

    fn try_enable_cache(&self) -> Result<(), Self::Err> {
        self.cache.enable();
        Ok(())
    }

    fn try_disable_cache(&self) -> Result<(), Self::Err> {
        self.cache.disable();
        self.try_empty_cache()
    }

    fn try_empty_cache(&self) -> Result<(), Self::Err> {
        #[cfg(not(feature = "no-std"))]
        let mut cache = self.cache.allocations.write().unwrap();
        #[cfg(feature = "no-std")]
        let mut cache = self.cache.allocations.write();
        for (&key, allocations) in cache.iter_mut() {
            assert!(key.num_bytes % key.size == 0);
            assert!(key.num_bytes < isize::MAX as usize);
            let len = key.num_bytes / key.size;
            let cap = len;
            for alloc in allocations.drain(..) {
                // SAFETY:
                // - "ptr must have been allocated using the global allocator, such as via the alloc::alloc function."
                //    - ✅ cpu uses global allocator
                // - "T needs to have the same alignment as what ptr was allocated with."
                //    - ✅ we are matching on the alignment below
                // - "The size of T times the capacity needs to be the same size as the pointer was allocated with."
                //    - ✅ covered by `key.num_bytes / key.size` and the `key.num_bytes % key.size == 0` assertion above
                // - "length needs to be less than or equal to capacity."
                //    - ✅ they are equal
                // - "The first length values must be properly initialized values of type T."
                //    - ✅ any bit pattern is valid for unsigned ints used below
                // - "capacity needs to be the capacity that the pointer was allocated with."
                //    - ✅ handled by assertion above (key.num_bytes % key.size == 0)
                // - "The allocated size in bytes must be no larger than isize::MAX. See the safety documentation of pointer::offset."
                //    - ✅ handled by assertion above
                debug_assert_eq!(std::alloc::Layout::new::<u8>().align(), 1);
                debug_assert_eq!(std::alloc::Layout::new::<u16>().align(), 2);
                debug_assert_eq!(std::alloc::Layout::new::<u32>().align(), 4);
                debug_assert_eq!(std::alloc::Layout::new::<u64>().align(), 8);
                match key.alignment {
                    1 => unsafe { drop(Vec::from_raw_parts(alloc.0 as *mut u8, len, cap)) },
                    2 => unsafe { drop(Vec::from_raw_parts(alloc.0 as *mut u16, len, cap)) },
                    4 => unsafe { drop(Vec::from_raw_parts(alloc.0 as *mut u32, len, cap)) },
                    8 => unsafe { drop(Vec::from_raw_parts(alloc.0 as *mut u64, len, cap)) },
                    _ => unreachable!(),
                };
            }
        }
        cache.clear();
        Ok(())
    }
}
