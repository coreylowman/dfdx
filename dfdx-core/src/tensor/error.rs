#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// Device is out of memory
    OutOfMemory,
    /// Not enough elements were provided when creating a tensor
    WrongNumElements,

    UnusedTensors(Vec<crate::tensor::UniqueId>),

    #[cfg(feature = "cuda")]
    CublasError(cudarc::cublas::result::CublasError),
    #[cfg(feature = "cuda")]
    CudaDriverError(cudarc::driver::DriverError),

    #[cfg(feature = "cudnn")]
    CudnnError(cudarc::cudnn::CudnnError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
