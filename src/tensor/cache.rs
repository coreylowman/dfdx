use std::{alloc::Layout, collections::BTreeMap, sync::RwLock};

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
            enabled: RwLock::new(true),
        }
    }
}

impl<Ptr> TensorCache<Ptr> {
    /// Returns `true` if the cache is enabled.
    pub(crate) fn is_enabled(&self) -> bool {
        *self.enabled.read().unwrap()
    }

    /// Disables the cache.
    pub(crate) fn disable(&self) {
        *self.enabled.write().unwrap() = false;
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
            num_bytes: num_bytes,
            size: layout.size(),
            alignment: layout.align(),
        };
        // Check if there is a cached allocation.
        let reuse = {
            let cache = self.allocations.read().unwrap();
            cache.contains_key(&key)
        };
        // If there is, remove it from the cache.
        // Otherwise, return `None`.
        if reuse {
            let mut cache = self.allocations.write().unwrap();
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
            num_bytes: num_bytes,
            size: layout.size(),
            alignment: layout.align(),
        };
        let mut cache = self.allocations.write().unwrap();
        if let std::collections::btree_map::Entry::Vacant(e) = cache.entry(key) {
            e.insert(std::vec![allocation]);
        } else {
            cache.get_mut(&key).unwrap().push(allocation);
        }
    }
}
