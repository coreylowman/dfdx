use std::{collections::BTreeMap, sync::RwLock};

/// A cache of allocations that can be reused.
///
/// The key is the number of bytes in the allocation.
/// The value is a list of allocations of that size.
///
/// The prescense of a key in the map, indicates that there is *at least one*
/// valid allocation. When the last value is removed from the list, the key
/// is removed.
#[derive(Debug)]
pub(crate) struct TensorCache<Ptr>(pub(crate) RwLock<BTreeMap<usize, Vec<Ptr>>>);

impl<Ptr> Default for TensorCache<Ptr> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<Ptr> TensorCache<Ptr> {
    /// Returns a cached allocation if one exists.
    /// Otherwise, returns `None`.
    pub(crate) fn try_pop(&self, num_bytes: usize) -> Option<Ptr> {
        // Check if there is a cached allocation.
        let reuse = {
            let cache = self.0.read().unwrap();
            cache.contains_key(&num_bytes)
        };
        // If there is, remove it from the cache.
        // Otherwise, return `None`.
        if reuse {
            let mut cache = self.0.write().unwrap();
            // unwrap is safe because we just checked for contains key above.
            let items = cache.get_mut(&num_bytes).unwrap();
            // unwrap is safe because reuse is only true if there's at least one item,
            // which is also maintained by the block directly below.
            let allocation = items.pop().unwrap();
            // If there are no more cached allocations of this size,
            // remove the entry from the cache.
            // This is important for correctness, because the presence
            // of an entry in the cache indicates that there are valid
            // allocations to use. (see `let reuse = { ... }` above).
            if items.is_empty() {
                cache.remove(&num_bytes);
            }
            Some(allocation)
        } else {
            None
        }
    }

    /// Inserts an allocation into the cache.
    pub(crate) fn insert(&self, num_bytes: usize, allocation: Ptr) {
        let mut cache = self.0.write().unwrap();
        if let std::collections::btree_map::Entry::Vacant(e) = cache.entry(num_bytes) {
            e.insert(std::vec![allocation]);
        } else {
            cache.get_mut(&num_bytes).unwrap().push(allocation);
        }
    }
}
