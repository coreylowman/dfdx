use std::{alloc::Layout, collections::BTreeMap, vec::Vec};

#[cfg(not(feature = "no-std"))]
use std::sync::RwLock;

#[cfg(feature = "no-std")]
use spin::RwLock;

/// A key for the tensor cache. Contains both number of bytes and informatino
/// about the layout of the allocation.
///
/// Since [Layout] doesn't impl Ord, we can't use it directly as a key
/// for a hasmap, meaning we need this extra datastructure. Otherwise
/// we could just using `(usize, Layout)` as the key.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct AllocationKey {
    /// The size of the allocation in bytes
    num_bytes: usize,
    /// The size of the type in bytes - from [Layout].
    size: usize,
    /// The alignment of the allocation in bytes - from [Layout].
    alignment: usize,
}

/// A cache of allocations that can be reused.
///
/// The key is the number of bytes in the allocation, AND the layout
/// that the allocation was created with. This is necessary for safely
/// reusing allocations, especially on the rust side of things, where the
/// allocator assumes memory is allocated & deallocated with the same layout.
/// The value is a list of allocations of that size.
///
/// The prescense of a key in the map, indicates that there is *at least one*
/// valid allocation. When the last value is removed from the list, the key
/// is removed.
#[derive(Debug)]
pub(crate) struct TensorCache<Ptr: CacheStorage> {
    allocations: RwLock<BTreeMap<AllocationKey, Vec<CacheWrapper<Ptr>>>>,
    enabled: RwLock<bool>,
}

pub(crate) trait CacheStorage: Sized {
    type Output<T>: CacheStorage;

    /// Unsafely converts the elements of a contiguous collection type to another type. Note:
    /// **This function is wildly unsafe**, see implementations for details
    unsafe fn transmute_elements<T>(self) -> Self::Output<T>;

    /// Uses transmute_elements to convert to an element type with alignment `align` before dropping.
    unsafe fn drop_with_alignment(self, align: usize) {
        match align {
            1 => drop(self.transmute_elements::<u8>()),
            2 => drop(self.transmute_elements::<u16>()),
            4 => drop(self.transmute_elements::<u32>()),
            8 => drop(self.transmute_elements::<u64>()),
            16 => drop(self.transmute_elements::<u128>()),
            _ => panic!("Invalid alignment"),
        }
    }
}

/// (Mostly) Safe wrapper around CacheStorage implementers
#[derive(Clone, Debug)]
struct CacheWrapper<Ptr: CacheStorage> {
    ptr: Option<Ptr>,
    alignment: usize,
    size: usize,
}

impl<Ptr: CacheStorage> Drop for CacheWrapper<Ptr> {
    fn drop(&mut self) {
        if let Some(ptr) = std::mem::take(&mut self.ptr) {
            unsafe { ptr.drop_with_alignment(self.alignment) }
        }
    }
}

impl<Ptr: CacheStorage> CacheWrapper<Ptr> {
    fn from_storage<T>(storage: Ptr::Output<T>) -> Self
    where
        Ptr::Output<T>: CacheStorage<Output<u8> = Ptr>,
    {
        let layout = Layout::new::<T>();
        Self {
            ptr: Some(unsafe { storage.transmute_elements::<u8>() }),
            alignment: layout.align(),
            size: layout.size(),
        }
    }

    fn check_key(&self, key: &AllocationKey) {
        assert_eq!(self.alignment, key.alignment, "Alignment does not match");
        assert_eq!(self.size, key.size, "Size does not match");
        // Implicitly assumes that T should not have any padding, but this should always be true of
        // primitive number types.
        assert_eq!(
            key.num_bytes % key.size,
            0,
            "Key is invalid or type is padded"
        );
    }

    // Safety: Same as slice.align_to, but considered safe internally
    // Produces storage containing uninitialized values
    fn into_storage<T>(mut self) -> Ptr::Output<T> {
        let layout = Layout::new::<T>();
        assert_eq!(layout.align(), self.alignment);
        assert_eq!(layout.size(), self.size);

        let ptr = std::mem::take(&mut self.ptr).unwrap();
        unsafe { ptr.transmute_elements() }
    }
}

impl<Ptr: CacheStorage> Default for TensorCache<Ptr> {
    fn default() -> Self {
        Self {
            allocations: Default::default(),
            enabled: RwLock::new(false),
        }
    }
}

impl<Ptr: CacheStorage> TensorCache<Ptr> {
    /// Returns the number of allocations in the cache.
    #[allow(unused)]
    pub(crate) fn len(&self) -> usize {
        #[cfg(not(feature = "no-std"))]
        {
            self.allocations.read().unwrap().len()
        }

        #[cfg(feature = "no-std")]
        {
            self.allocations.read().len()
        }
    }

    /// Returns `true` if the cache is enabled.
    pub(crate) fn is_enabled(&self) -> bool {
        #[cfg(not(feature = "no-std"))]
        {
            *self.enabled.read().unwrap()
        }
        #[cfg(feature = "no-std")]
        {
            *self.enabled.read()
        }
    }

    /// Enables the cache.
    pub(crate) fn enable(&self) {
        #[cfg(not(feature = "no-std"))]
        {
            *self.enabled.write().unwrap() = true;
        }

        #[cfg(feature = "no-std")]
        {
            *self.enabled.write() = true;
        }
    }

    /// Disables the cache.
    pub(crate) fn disable(&self) {
        #[cfg(not(feature = "no-std"))]
        {
            *self.enabled.write().unwrap() = false;
        }

        #[cfg(feature = "no-std")]
        {
            *self.enabled.write() = false;
        }
    }

    /// Returns a cached allocation if one exists.
    /// Otherwise, returns `None`.
    pub(crate) fn try_pop<E>(&self, len: usize) -> Option<Ptr::Output<E>> {
        if !self.is_enabled() {
            return None;
        }

        let layout = Layout::new::<E>();
        let num_bytes = len * std::mem::size_of::<E>();
        let key = AllocationKey {
            num_bytes,
            size: layout.size(),
            alignment: layout.align(),
        };
        // Check if there is a cached allocation.
        let reuse = {
            #[cfg(not(feature = "no-std"))]
            let cache = self.allocations.read().unwrap();
            #[cfg(feature = "no-std")]
            let cache = self.allocations.read();
            cache.contains_key(&key)
        };
        // If there is, remove it from the cache.
        // Otherwise, return `None`.
        if reuse {
            #[cfg(not(feature = "no-std"))]
            let mut cache = self.allocations.write().unwrap();
            #[cfg(feature = "no-std")]
            let mut cache = self.allocations.write();
            // unwrap is safe because we just checked for contains key above.
            let items = cache.get_mut(&key).unwrap();
            // unwrap is safe because reuse is only true if there's at least one item,
            // which is also maintained by the block directly below.
            let allocation = items.pop().unwrap();
            // If there are no more cached allocations of this size,
            // remove the entry from the cache.
            // This is important for correctness, because the presence
            // of an entry in the cache indicates that there are valid
            // allocations to use. (see `let reuse = { ... }` above).
            if items.is_empty() {
                cache.remove(&key);
            }
            allocation.check_key(&key);
            Some(allocation.into_storage())
        } else {
            None
        }
    }

    /// Inserts an allocation into the cache.
    pub(crate) fn insert<E>(&self, len: usize, allocation: Ptr::Output<E>)
    where
        Ptr::Output<E>: CacheStorage<Output<u8> = Ptr>,
    {
        if !self.is_enabled() {
            return;
        }

        let allocation = CacheWrapper::from_storage(allocation);
        let layout = Layout::new::<E>();
        let num_bytes = len * layout.size();
        let key = AllocationKey {
            num_bytes,
            size: layout.size(),
            alignment: layout.align(),
        };
        allocation.check_key(&key);
        #[cfg(not(feature = "no-std"))]
        let mut cache = self.allocations.write().unwrap();
        #[cfg(feature = "no-std")]
        let mut cache = self.allocations.write();
        if let std::collections::btree_map::Entry::Vacant(e) = cache.entry(key) {
            #[cfg(not(feature = "no-std"))]
            {
                e.insert(std::vec![allocation]);
            }
            #[cfg(feature = "no-std")]
            {
                let mut allocations = Vec::new();
                allocations.push(allocation);
                e.insert(allocations);
            }
        } else {
            cache.get_mut(&key).unwrap().push(allocation);
        }
    }

    pub(crate) fn clear(&self) {
        #[cfg(not(feature = "no-std"))]
        self.allocations.write().unwrap().clear();
        #[cfg(feature = "no-std")]
        self.allocations.write().clear();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_try_pop_on_disabled_cache() {
        let cache: TensorCache<Vec<u8>> = Default::default();
        cache.enable();
        assert!(cache.is_enabled());
        cache.disable();
        assert!(!cache.is_enabled());
        assert_eq!(cache.try_pop::<f32>(1), None);
        assert_eq!(cache.try_pop::<f32>(1), None);
    }

    #[test]
    fn test_try_pop_on_empty_cache() {
        let cache: TensorCache<Vec<u8>> = Default::default();
        cache.enable();
        assert_eq!(cache.try_pop::<f32>(1), None);
        assert_eq!(cache.try_pop::<f32>(1), None);
    }

    #[test]
    fn test_try_pop_on_cache_with_multiple_sizes_and_alignment() {
        let cache: TensorCache<Vec<u8>> = Default::default();
        cache.enable();
        cache.insert::<f32>(1, vec![0.0]);
        cache.insert::<f32>(1, vec![1.0]);
        cache.insert::<f32>(1, vec![2.0]);
        cache.insert::<f32>(2, vec![3.0; 2]);
        cache.insert::<f32>(2, vec![4.0; 2]);
        cache.insert::<f32>(2, vec![5.0; 2]);
        cache.insert::<f64>(1, vec![6.0]);
        cache.insert::<f64>(1, vec![7.0]);
        cache.insert::<f64>(1, vec![8.0]);
        cache.insert::<f64>(2, vec![9.0; 2]);
        cache.insert::<f64>(2, vec![10.0; 2]);
        cache.insert::<f64>(2, vec![11.0; 2]);
        assert_eq!(cache.try_pop::<f32>(1), Some(vec![2.0]));
        assert_eq!(cache.try_pop::<f32>(1), Some(vec![1.0]));
        assert_eq!(cache.try_pop::<f32>(1), Some(vec![0.0]));
        assert_eq!(cache.try_pop::<f32>(1), None);
        assert_eq!(cache.try_pop::<f32>(2), Some(vec![5.0; 2]));
        assert_eq!(cache.try_pop::<f32>(2), Some(vec![4.0; 2]));
        assert_eq!(cache.try_pop::<f32>(2), Some(vec![3.0; 2]));
        assert_eq!(cache.try_pop::<f32>(2), None);
        assert_eq!(cache.try_pop::<f64>(1), Some(vec![8.0]));
        assert_eq!(cache.try_pop::<f64>(1), Some(vec![7.0]));
        assert_eq!(cache.try_pop::<f64>(1), Some(vec![6.0]));
        assert_eq!(cache.try_pop::<f64>(1), None);
        assert_eq!(cache.try_pop::<f64>(2), Some(vec![11.0; 2]));
        assert_eq!(cache.try_pop::<f64>(2), Some(vec![10.0; 2]));
        assert_eq!(cache.try_pop::<f64>(2), Some(vec![9.0; 2]));
        assert_eq!(cache.try_pop::<f64>(2), None);
    }
}
