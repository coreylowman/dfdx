mod allocate;
mod device;
mod index;
mod iterate;
mod views;

pub(crate) use device::StridedArray;
pub(crate) use iterate::LendingIterator;
pub(crate) use views::{View, ViewMut};

pub use device::{Cpu, CpuError};
