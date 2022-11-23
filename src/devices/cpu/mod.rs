mod allocate;
mod device;
mod index;
mod iterate;
mod kernels;
mod views;

pub(crate) use iterate::LendingIterator;
pub(crate) use views::{View, ViewMut};

pub(crate) use device::StridedArray;

pub use device::{Cpu, CpuError};
