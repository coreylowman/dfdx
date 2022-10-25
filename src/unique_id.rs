//! A simple implementation of a UID used as a unique key for tensors.

/// An id used in to associate gradients with Tensors.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub struct UniqueId(usize);

/// Generate a [UniqueId].
pub(crate) fn unique_id() -> UniqueId {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    UniqueId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

impl UniqueId {
    pub(crate) fn as_u64(&self) -> u64 {
        self.0 as u64
    }
}

/// Something that has a [UniqueId]
pub trait HasUniqueId {
    fn id(&self) -> &UniqueId;
}

/// Internal only - for resetting ids of tensor
pub(crate) mod internal {
    /// Internal only - for resetting ids of tensor
    pub trait ResetId {
        /// Internal only - for resetting ids of tensor
        fn reset_id(&mut self);
    }
}
