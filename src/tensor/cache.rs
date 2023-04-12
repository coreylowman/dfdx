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
pub(crate) struct AllocationKey {
    pub num_bytes: usize,
    /// The size of the allocation in bytes - from [Layout].
    pub size: usize,
    /// The alignment of the allocation in bytes - from [Layout].
    pub alignment: usize,
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
pub(crate) struct TensorCache<Ptr> {
    pub(crate) allocations: RwLock<BTreeMap<AllocationKey, Vec<Ptr>>>,
    pub(crate) enabled: RwLock<bool>,
}

impl<Ptr> Default for TensorCache<Ptr> {
    fn default() -> Self {
        Self {
            allocations: Default::default(),
            enabled: RwLock::new(false),
        }
    }
}

impl<Ptr> TensorCache<Ptr> {
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

    /// Disables the cache.
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
    pub(crate) fn try_pop<E>(&self, len: usize) -> Option<Ptr> {
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
            Some(allocation)
        } else {
            None
        }
    }

    /// Inserts an allocation into the cache.
    pub(crate) fn insert<E>(&self, len: usize, allocation: Ptr) {
        if !self.is_enabled() {
            // This is a panic because it's a bug in the library.
            panic!("Tried to insert into a disabled cache.");
        }

        let layout = Layout::new::<E>();
        let num_bytes = len * std::mem::size_of::<E>();
        let key = AllocationKey {
            num_bytes,
            size: layout.size(),
            alignment: layout.align(),
        };
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
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[should_panic(expected = "Tried to insert into a disabled cache.")]
    fn test_insert_on_disabled_cache() {
        let cache: TensorCache<usize> = Default::default();
        cache.insert::<f32>(1, 0);
    }

    #[test]
    fn test_try_pop_on_disabled_cache() {
        let cache: TensorCache<usize> = Default::default();
        cache.enable();
        assert!(cache.is_enabled());
        cache.disable();
        assert!(!cache.is_enabled());
        assert_eq!(cache.try_pop::<f32>(1), None);
        assert_eq!(cache.try_pop::<f32>(1), None);
    }

    #[test]
    fn test_try_pop_on_empty_cache() {
        let cache: TensorCache<usize> = Default::default();
        cache.enable();
        assert_eq!(cache.try_pop::<f32>(1), None);
        assert_eq!(cache.try_pop::<f32>(1), None);
    }

    #[test]
    fn test_try_pop_on_cache_with_multiple_sizes_and_alignment() {
        let cache: TensorCache<usize> = Default::default();
        cache.enable();
        cache.insert::<f32>(1, 0);
        cache.insert::<f32>(1, 1);
        cache.insert::<f32>(1, 2);
        cache.insert::<f32>(2, 3);
        cache.insert::<f32>(2, 4);
        cache.insert::<f32>(2, 5);
        cache.insert::<f64>(1, 6);
        cache.insert::<f64>(1, 7);
        cache.insert::<f64>(1, 8);
        cache.insert::<f64>(2, 9);
        cache.insert::<f64>(2, 10);
        cache.insert::<f64>(2, 11);
        assert_eq!(cache.try_pop::<f32>(1), Some(2));
        assert_eq!(cache.try_pop::<f32>(1), Some(1));
        assert_eq!(cache.try_pop::<f32>(1), Some(0));
        assert_eq!(cache.try_pop::<f32>(1), None);
        assert_eq!(cache.try_pop::<f32>(2), Some(5));
        assert_eq!(cache.try_pop::<f32>(2), Some(4));
        assert_eq!(cache.try_pop::<f32>(2), Some(3));
        assert_eq!(cache.try_pop::<f32>(2), None);
        assert_eq!(cache.try_pop::<f64>(1), Some(8));
        assert_eq!(cache.try_pop::<f64>(1), Some(7));
        assert_eq!(cache.try_pop::<f64>(1), Some(6));
        assert_eq!(cache.try_pop::<f64>(1), None);
        assert_eq!(cache.try_pop::<f64>(2), Some(11));
        assert_eq!(cache.try_pop::<f64>(2), Some(10));
        assert_eq!(cache.try_pop::<f64>(2), Some(9));
        assert_eq!(cache.try_pop::<f64>(2), None);
    }
}
