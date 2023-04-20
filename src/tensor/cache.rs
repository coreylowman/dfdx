use std::{
    alloc::Layout,
    collections::{BTreeMap, VecDeque},
    vec::Vec,
};

#[cfg(not(feature = "no-std"))]
use std::vec;

#[cfg(feature = "no-std")]
use alloc::vec;

#[cfg(not(feature = "no-std"))]
use std::sync::RwLock;

#[cfg(feature = "no-std")]
use spin::RwLock;

macro_rules! read {
    ($x:expr) => {{
        #[cfg(not(feature = "no-std"))]
        {
            $x.read().unwrap()
        }
        #[cfg(feature = "no-std")]
        {
            $x.read()
        }
    }};
}

macro_rules! write {
    ($x:expr) => {{
        #[cfg(not(feature = "no-std"))]
        {
            $x.write().unwrap()
        }
        #[cfg(feature = "no-std")]
        {
            $x.write()
        }
    }};
}

/// A key for the tensor cache. Contains both number of bytes and information
/// about the layout of the allocation.
///
/// Since [Layout] doesn't impl Ord, we can't use it directly as a key
/// for a hashmap, meaning we need this extra datastructure. Otherwise
/// we could just using `(usize, Layout)` as the key.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
struct AllocationKey {
    /// The size of the allocation in bytes
    num_bytes: usize,
    /// The size of the type in bytes - from [Layout].
    size: usize,
    /// The alignment of the allocation in bytes - from [Layout].
    alignment: usize,
}

#[derive(Debug)]
struct AllocationGroup<Ptr: CacheStorage> {
    // Tracks the number of matching 'AllocationKey's in drop_queue to ignore. This is used to
    // "remove" the next instance of the matching AllocationKey in the drop_queue, without having
    // to run an O(n)
    ignore_drops: usize,
    allocations: Vec<CacheWrapper<Ptr>>,
}

/// A cache of allocations that can be reused.
///
/// The key is the number of bytes in the allocation, AND the layout
/// that the allocation was created with. This is necessary for safely
/// reusing allocations, especially on the rust side of things, where the
/// allocator assumes memory is allocated & deallocated with the same layout.
/// The value is a list of allocations of that size.
///
/// The presence of a key in the map, indicates that there is *at least one*
/// valid allocation. When the last value is removed from the list, the key
/// is removed.
///
/// Constraint: for a given value of AllocationKey, the following must hold:
///
/// (instances in drop_queue) = (group.ignore_drops) + (group.allocations.len())
#[derive(Debug)]
pub(crate) struct TensorCache<Ptr: CacheStorage> {
    allocations: RwLock<BTreeMap<AllocationKey, AllocationGroup<Ptr>>>,
    enabled: RwLock<bool>,

    drop_queue: RwLock<VecDeque<AllocationKey>>,
    size: RwLock<usize>,
    max_size: RwLock<usize>,
}

pub(crate) trait CacheStorage: Sized {
    type Output<T>: CacheStorage;

    /// returns the allocations's size in bytes
    fn size(&self) -> usize;

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
#[derive(Debug)]
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
        // Implicitly assumes that T does not have any padding, but this should always be true of
        // primitive number types.
        assert_eq!(
            key.num_bytes % key.size,
            0,
            "Key is invalid or type is padded"
        );
    }

    fn size(&self) -> usize {
        self.ptr.as_ref().unwrap().size()
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

impl<Ptr: CacheStorage> AllocationGroup<Ptr> {
    fn is_empty(&self) -> bool {
        self.allocations.is_empty() && self.ignore_drops == 0
    }
}

impl<Ptr: CacheStorage> Default for TensorCache<Ptr> {
    fn default() -> Self {
        Self {
            allocations: Default::default(),
            enabled: RwLock::new(false),
            drop_queue: Default::default(),
            size: RwLock::new(0),
            max_size: RwLock::new(0),
        }
    }
}

impl<Ptr: CacheStorage> TensorCache<Ptr> {
    /// Returns the number of allocations in the cache.
    #[allow(unused)]
    pub(crate) fn len(&self) -> usize {
        read!(self.allocations)
            .values()
            .map(|group| group.allocations.len())
            .sum()
    }

    /// Returns the number of bytes occupied by allocations in the cache.
    #[allow(unused)]
    pub(crate) fn size(&self) -> usize {
        *read!(self.size)
    }

    /// Returns `true` if the cache is enabled.
    pub(crate) fn is_enabled(&self) -> bool {
        *read!(self.enabled)
    }

    /// Enables the cache.
    pub(crate) fn enable(&self, size: usize) {
        *write!(self.enabled) = true;
        *write!(self.max_size) = size;
    }

    /// Disables the cache.
    pub(crate) fn disable(&self) {
        *write!(self.enabled) = false;
    }

    /// Sets the maximum size of the cache
    #[allow(unused)]
    pub(crate) fn set_max_size(&self, size: usize) {
        *write!(self.max_size) = size;

        if size < *read!(self.size) {
            self.shrink();
        }
    }

    /// Shrinks the cache so its buffers contain at most `max_size` bytes
    fn shrink(&self) {
        let mut size = write!(self.size);
        let max_size = read!(self.max_size);
        let mut drop_queue = write!(self.drop_queue);
        let mut allocations = write!(self.allocations);

        debug_assert_eq!(
            *size,
            allocations
                .values()
                .flat_map(|group| &group.allocations)
                .map(|alloc| alloc.size())
                .sum::<usize>()
        );

        while *size > *max_size {
            let key = drop_queue
                .pop_front()
                .expect("ignore_drops values were set too high");
            let Some(alloc_group) = allocations.get_mut(&key) else { continue };

            if alloc_group.ignore_drops > 0 {
                alloc_group.ignore_drops -= 1;
            } else {
                let allocation = alloc_group
                    .allocations
                    .pop()
                    .expect("ignore_drops values were set too low");
                if alloc_group.is_empty() {
                    allocations.remove(&key);
                }
                *size -= allocation.size();
            }
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
        let reuse = read!(self.allocations)
            .get(&key)
            .map_or(false, |group| !group.allocations.is_empty());
        // If there is, remove it from the cache.
        // Otherwise, return `None`.
        if reuse {
            let mut cache = write!(self.allocations);
            // unwrap is safe because we just checked for contains key above.
            let items = cache.get_mut(&key).unwrap();
            items.ignore_drops += 1;
            // unwrap is safe because reuse is only true if there's at least one item,
            // which is also maintained by the block directly below.
            let allocation = items.allocations.pop().unwrap();
            allocation.check_key(&key);
            *write!(self.size) -= allocation.size();
            Some(allocation.into_storage())
        } else {
            None
        }
    }

    /// Inserts an allocation into the cache.
    pub(crate) fn insert<E>(&self, allocation: Ptr::Output<E>)
    where
        Ptr::Output<E>: CacheStorage<Output<u8> = Ptr>,
    {
        if !self.is_enabled() {
            return;
        }

        let allocation = CacheWrapper::from_storage(allocation);
        let layout = Layout::new::<E>();
        let num_bytes = allocation.size();
        let key = AllocationKey {
            num_bytes,
            size: layout.size(),
            alignment: layout.align(),
        };
        allocation.check_key(&key);
        *write!(self.size) += allocation.size();
        write!(self.drop_queue).push_back(key);
        let mut cache = write!(self.allocations);
        if let std::collections::btree_map::Entry::Vacant(e) = cache.entry(key) {
            e.insert(AllocationGroup {
                allocations: vec![allocation],
                ignore_drops: 0,
            });
        } else {
            cache.get_mut(&key).unwrap().allocations.push(allocation);
        }
        std::mem::drop(cache);
        self.shrink();
    }

    pub(crate) fn clear(&self) {
        write!(self.allocations).clear();
        write!(self.drop_queue).clear();
        *write!(self.size) = 0;
    }

    #[allow(unused)]
    fn clear_check(&self) {
        self.set_max_size(0);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_try_pop_on_disabled_cache() {
        let cache: TensorCache<Vec<u8>> = Default::default();
        cache.enable(1000);
        assert!(cache.is_enabled());
        cache.disable();
        assert!(!cache.is_enabled());
        assert_eq!(cache.try_pop::<f32>(1), None);
        assert_eq!(cache.try_pop::<f32>(1), None);
    }

    #[test]
    fn test_try_pop_on_empty_cache() {
        let cache: TensorCache<Vec<u8>> = Default::default();
        cache.enable(1000);
        assert_eq!(cache.try_pop::<f32>(1), None);
        assert_eq!(cache.try_pop::<f32>(1), None);
    }

    #[test]
    fn test_try_pop_on_cache_with_multiple_sizes_and_alignment() {
        let cache: TensorCache<Vec<u8>> = Default::default();
        cache.enable(1000);
        cache.insert::<f32>(vec![0.0]);
        cache.insert::<f32>(vec![1.0]);
        cache.insert::<f32>(vec![2.0]);
        cache.insert::<f32>(vec![3.0; 2]);
        cache.insert::<f32>(vec![4.0; 2]);
        cache.insert::<f32>(vec![5.0; 2]);
        cache.insert::<f64>(vec![6.0]);
        cache.insert::<f64>(vec![7.0]);
        cache.insert::<f64>(vec![8.0]);
        cache.insert::<f64>(vec![9.0; 2]);
        cache.insert::<f64>(vec![10.0; 2]);
        cache.insert::<f64>(vec![11.0; 2]);
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
        cache.clear_check();
    }

    #[test]
    fn test_shrink() {
        let cache: TensorCache<Vec<u8>> = Default::default();
        cache.enable(16);
        cache.insert::<u8>(vec![1; 1]);
        cache.insert::<u8>(vec![2; 1]);
        cache.insert::<u8>(vec![1; 2]);
        cache.insert::<u8>(vec![1; 4]);
        cache.insert::<u8>(vec![1; 8]);
        assert_eq!(cache.len(), 5);
        assert_eq!(cache.size(), 16);
        cache.insert::<u8>(vec![2; 8]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.size(), 16);
        cache.insert::<u8>(vec![3; 1]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.size(), 9);
        cache.insert::<u8>(vec![1; 12]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.size(), 13);
        cache.clear_check();
    }

    #[test]
    fn test_pop_and_shrink() {
        let cache: TensorCache<Vec<u8>> = Default::default();
        cache.enable(16);
        cache.insert::<u8>(vec![1; 1]);
        cache.insert::<u8>(vec![2; 1]);
        cache.insert::<u8>(vec![1; 2]);
        cache.insert::<u8>(vec![1; 4]);
        cache.insert::<u8>(vec![1; 8]);
        assert_eq!(cache.len(), 5);
        assert_eq!(cache.size(), 16);

        assert_eq!(cache.try_pop::<u8>(1), Some(vec![2]));
        assert_eq!(cache.try_pop::<u8>(2), Some(vec![1; 2]));
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.size(), 13);

        cache.insert::<u8>(vec![2; 8]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.size(), 16);

        assert_eq!(cache.try_pop::<u8>(8), Some(vec![2; 8]));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.size(), 8);

        cache.insert::<u8>(vec![2; 4]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.size(), 12);

        cache.clear_check();
    }
}
