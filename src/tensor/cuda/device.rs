use crate::shapes::{Shape, Unit};
use crate::tensor::cpu::{Cpu, CpuError, NdIndex};
use crate::tensor::{DeviceStorage, HasErr, Tensor};

use cudarc::cudnn::result::CudnnError;
use cudarc::cudnn::Cudnn;
use cudarc::{
    cublas::{result::CublasError, CudaBlas},
    driver::{CudaDevice, CudaSlice, CudaStream, DeviceSlice, DriverError},
};

use std::sync::MutexGuard;
use std::{
    sync::{Arc, Mutex},
    vec::Vec,
};

/// A Cuda device that enables constructing tensors on GPUs
/// & running GPU kernels.
#[derive(Clone, Debug)]
pub struct Cuda {
    pub(crate) cpu: Cpu,
    pub(crate) dev: Arc<CudaDevice>,
    pub(crate) blas: Arc<CudaBlas>,
    pub(crate) cudnn: Arc<Cudnn>,
    /// A second stream for kernels to optionally execute on.
    pub(crate) par_stream: Arc<CudaStream>,
    pub(crate) workspace: Arc<Mutex<CudaSlice<u8>>>,
}

#[derive(Debug)]
pub enum CudaError {
    Blas(CublasError),
    Cudnn(CudnnError),
    Driver(DriverError),
    Cpu(CpuError),
}

impl From<CpuError> for CudaError {
    fn from(value: CpuError) -> Self {
        Self::Cpu(value)
    }
}

impl From<CublasError> for CudaError {
    fn from(value: CublasError) -> Self {
        Self::Blas(value)
    }
}

impl From<DriverError> for CudaError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

impl From<CudnnError> for CudaError {
    fn from(value: CudnnError) -> Self {
        Self::Cudnn(value)
    }
}

impl Default for Cuda {
    fn default() -> Self {
        Self::seed_from_u64(0)
    }
}

impl Cuda {
    /// Constructs rng with the given seed.
    pub fn seed_from_u64(seed: u64) -> Self {
        Self::try_seed_from_u64(seed).unwrap()
    }

    /// Constructs rng with the given seed.
    pub fn try_seed_from_u64(seed: u64) -> Result<Self, CudaError> {
        Self::try_build(0, seed)
    }

    /// Constructs with the given seed & device ordinal
    pub fn try_build(ordinal: usize, seed: u64) -> Result<Self, CudaError> {
        let cpu = Cpu::seed_from_u64(seed);
        let dev = CudaDevice::new(ordinal)?;
        let blas = Arc::new(CudaBlas::new(dev.clone())?);
        let cudnn = Arc::new(Cudnn::new(dev.clone())?);
        let par_stream = Arc::new(dev.fork_default_stream()?);
        let workspace = Arc::new(Mutex::new(dev.alloc_zeros::<u8>(0)?));
        Ok(Self {
            cpu,
            dev,
            blas,
            cudnn,
            par_stream,
            workspace,
        })
    }
}

impl Cuda {
    #[allow(unused)]
    pub(crate) unsafe fn get_workspace<E>(
        &self,
        len: usize,
    ) -> Result<MutexGuard<CudaSlice<u8>>, CudaError> {
        let num_bytes_required = len * std::mem::size_of::<E>();
        let mut workspace = self.workspace.as_ref().lock().unwrap();

        // re-allocate a larger workspace
        if workspace.num_bytes() < num_bytes_required {
            // we are about to memset this to zero, so this is still okay
            *workspace = unsafe { self.dev.alloc::<u8>(num_bytes_required) }?;
        }

        Ok(workspace)
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl HasErr for Cuda {
    type Err = CudaError;
}

impl DeviceStorage for Cuda {
    type Vec<E: Unit> = CudaSlice<E>;

    fn try_alloc_grad<E: Unit>(&self, other: &Self::Vec<E>) -> Result<Self::Vec<E>, Self::Err> {
        let grad = self.dev.alloc_zeros(other.len())?;
        Ok(grad)
    }

    fn random_u64(&self) -> u64 {
        self.cpu.random_u64()
    }

    fn tensor_to_vec<S: Shape, E: Unit, T>(&self, tensor: &Tensor<S, E, Self, T>) -> Vec<E> {
        let buf: Vec<E> = tensor.data.try_clone().unwrap().try_into().unwrap();
        debug_assert_eq!(buf.len(), tensor.data.len());
        let mut idx = NdIndex::new(tensor.shape, tensor.strides);
        let mut contiguous = Vec::with_capacity(tensor.shape.num_elements());
        while let Some(i) = idx.next() {
            contiguous.push(buf[i]);
        }
        contiguous
    }

    fn try_synchronize(&self) -> Result<(), CudaError> {
        self.dev.synchronize().map_err(CudaError::from)
    }
}
