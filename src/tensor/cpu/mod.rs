mod allocate;
mod device;
mod index;
mod iterate;

pub(crate) use index::index_to_i;
pub(crate) use iterate::{LendingIterator, NdIndex};

pub use device::{Cpu, CpuError};
