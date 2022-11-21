mod allocate;
mod device;
mod index;
mod iterate;
mod kernels;

pub(crate) use device::StridedArray;
use iterate::LendingIterator;

pub use device::{Cpu, CpuError};
