use crate::shapes::{Shape, Unit};
use crate::tensor::cpu::{Cpu, CpuError, NdIndex};
use crate::tensor::{DeviceStorage, HasErr, Tensor};

use cudarc::{
    cublas::{result::CublasError, CudaBlas},
    driver::{CudaDevice, CudaSlice, DeviceSlice, DriverError},
};
use std::{sync::Arc, vec::Vec};

#[derive(Clone, Debug)]
pub struct Cuda {
    pub(crate) cpu: Cpu,
    pub(crate) dev: Arc<CudaDevice>,
    pub(crate) blas: Arc<CudaBlas>,
}

#[derive(Debug)]
pub enum CudaError {
    Blas(CublasError),
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

    pub fn try_build(ordinal: usize, seed: u64) -> Result<Self, CudaError> {
        let cpu = Cpu::seed_from_u64(seed);
        let dev = CudaDevice::new(ordinal)?;
        let blas = Arc::new(CudaBlas::new(dev.clone())?);
        Ok(Self { cpu, dev, blas })
    }

    /// Block until kernels finish processing. Useful for benchmarking.
    ///
    /// Examples:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let dev: Cuda = Default::default();
    /// let a = dev.tensor([1., 2., 3.]);
    /// let _b = a.square();
    /// dev.synchronize().unwrap(); // blocks until square kernel finishes.
    /// ```
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.dev.synchronize().map_err(CudaError::from)
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
}
