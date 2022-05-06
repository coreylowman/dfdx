#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct UniqueId(pub(crate) usize);

pub(crate) fn unique_id() -> UniqueId {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    UniqueId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

impl std::ops::Deref for UniqueId {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait HasUniqueId {
    fn id(&self) -> &UniqueId;
}
