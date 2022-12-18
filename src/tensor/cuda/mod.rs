mod allocate;
mod device;

pub(crate) use device::CudaArray;

pub use device::{Cuda, CudaError};
