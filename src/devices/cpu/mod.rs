mod allocate;
mod device;
mod index;
mod iterate;
mod kernels;
mod views;

use iterate::LendingIterator;
use views::{View, ViewMut};

pub(crate) use device::StridedArray;

pub use device::{Cpu, CpuError};
