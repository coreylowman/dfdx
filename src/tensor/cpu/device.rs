use crate::shapes::{Shape, Unit};
use crate::tensor::{
    cache::{CacheStorage, TensorCache},
    cpu::LendingIterator,
    storage_traits::*,
    Tensor,
};
use core::alloc::Layout;
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

impl<T> CacheStorage for Vec<T> {
    type Output<T2> = Vec<T2>;

    fn size(&self) -> usize {
        // size in bytes of the underlying allocation
        Layout::array::<T>(self.len()).unwrap().size()
    }

    /// Unsafely converts the elements of a vector to a new type.
    ///
    /// # Safety
    ///
    /// * Has all of the potential pitfalls of slice.align_to
    /// * If converting to a type with a different alignment, the caller must convert back to a
    /// type with the same alignment before dropping
    /// * If converting to a type with a different alignment, the caller must not grow or shrink
    /// the allocation of the returned vector
    unsafe fn transmute_elements<T2>(mut self) -> Self::Output<T2> {
        let src_layout = Layout::new::<T>().pad_to_align();
        let dst_layout = Layout::new::<T2>().pad_to_align();

        let byte_len = self.len() * src_layout.size();
        let byte_capacity = self.capacity() * src_layout.size();
        let ptr = self.as_mut_ptr();
        std::mem::forget(self);

        let dst_size = dst_layout.size();

        assert_eq!(
            ptr.align_offset(dst_layout.align()),
            0,
            "Allocation is improperly aligned"
        );
        assert_eq!(byte_len % dst_size, 0, "Length is improperly sized");
        assert_eq!(
            byte_capacity % dst_size,
            0,
            "Allocation is improperly sized"
        );

        let len = byte_len / dst_size;
        let capacity = byte_capacity / dst_size;

        // Safety:
        // * T2 may not have the same alignment as the initial vector, it is the caller's
        // responsiblity to ensure that the vector is converted to a type with the correct
        // alignment before dropping
        // * The first len values may not be correctly initialized, it is the caller's
        // responsibility to ensure correct values before usage
        //
        // * ptr is allocated with the global allocator as long as self was
        // * length is less than or equal to capacity as long as this is true of self.
        // * capacity is the capacity the pointer was allocated with as long as this is true of
        // self
        // * The allocated size is less than isize::MAX as long as this is true of self
        Vec::from_raw_parts(ptr as *mut T2, len, capacity)
    }
}

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
    pub(crate) cache: Arc<TensorCache<Vec<u8>>>,
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
    pub(crate) cache: Arc<TensorCache<Vec<u8>>>,
}

impl<E: Clone> Clone for CachableVec<E> {
    fn clone(&self) -> Self {
        let numel = self.data.len();
        self.cache.try_pop::<E>(numel).map_or_else(
            || Self {
                data: self.data.clone(),
                cache: self.cache.clone(),
            },
            |mut data| {
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
        let data = std::mem::take(&mut self.data);
        self.cache.insert::<E>(data);
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

    fn try_enable_cache(&self, size: usize) -> Result<(), Self::Err> {
        self.cache.enable(size);
        Ok(())
    }

    fn try_disable_cache(&self) -> Result<(), Self::Err> {
        self.cache.disable();
        self.try_empty_cache()
    }

    fn try_empty_cache(&self) -> Result<(), Self::Err> {
        self.cache.clear();
        Ok(())
    }

    fn try_set_cache_size(&self, size: usize) -> Result<(), Self::Err> {
        self.cache.set_max_size(size);
        Ok(())
    }
}
