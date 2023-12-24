/// Represents a number of different errors that can occur from creating tensors
/// or launching tensor operations. This encompasses both Cpu and CUDA errors.
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// Device is out of memory
    OutOfMemory,
    /// Not enough elements were provided when creating a tensor
    WrongNumElements,
    /// Some tensors were unused by an optimizer in a graph.
    UnusedTensors(std::vec::Vec<crate::tensor::UniqueId>),
    #[cfg(feature = "cuda")]
    CublasError(cudarc::cublas::result::CublasError),
    #[cfg(feature = "cuda")]
    CudaDriverError(cudarc::driver::DriverError),

    #[cfg(feature = "cudnn")]
    CudnnError(cudarc::cudnn::CudnnError),

    #[cfg(feature = "webgpu")]
    WebgpuAdapterNotFound,

    #[cfg(feature = "webgpu")]
    WebgpuRequestDeviceError(wgpu::RequestDeviceError),

    #[cfg(feature = "webgpu")]
    WebgpuSourceLoadError,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
