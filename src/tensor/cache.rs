use std::{collections::BTreeMap, sync::RwLock};

#[derive(Debug)]
pub(crate) struct TensorCache<Ptr>(pub(crate) RwLock<BTreeMap<usize, Vec<Ptr>>>);

impl<Ptr> Default for TensorCache<Ptr> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<Ptr> TensorCache<Ptr> {
    pub(crate) fn try_pop(&self, num_bytes: usize) -> Option<Ptr> {
        let reuse = {
            let cache = self.0.read().unwrap();
            cache.contains_key(&num_bytes)
        };
        if reuse {
            let mut cache = self.0.write().unwrap();
            let items = cache.get_mut(&num_bytes).unwrap();
            let allocation = items.pop().unwrap();
            if items.is_empty() {
                cache.remove(&num_bytes);
            }
            Some(allocation)
        } else {
            None
        }
    }

    pub(crate) fn insert(&self, num_bytes: usize, allocation: Ptr) {
        let mut cache = self.0.write().unwrap();
        if let std::collections::btree_map::Entry::Vacant(e) = cache.entry(num_bytes) {
            e.insert(std::vec![allocation]);
        } else {
            cache.get_mut(&num_bytes).unwrap().push(allocation);
        }
    }
}
