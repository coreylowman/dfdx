use crate::shapes::{Dtype, HasDtype, HasShape, HasUnitType, Shape, Unit};
use crate::tensor::storage_traits::{DeviceStorage, HasErr};

use cudarc::{
    blas::CudaBlas,
    cublas::result::CublasError,
    device::{BuildError, CudaDevice, CudaDeviceBuilder, CudaSlice},
    driver::result::DriverError,
};
use rand::SeedableRng;
use rand::{rngs::StdRng, Rng};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub enum CudaError {
    Build(BuildError),
    Blas(CublasError),
    Driver(DriverError),
}

impl From<BuildError> for CudaError {
    fn from(value: BuildError) -> Self {
        Self::Build(value)
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

#[derive(Clone, Debug)]
pub struct Cuda {
    pub(crate) rng: Arc<Mutex<StdRng>>,
    pub(crate) dev: Arc<CudaDevice>,
    pub(crate) blas: Arc<CudaBlas>,
}

impl Default for Cuda {
    fn default() -> Self {
        Self::seed_from_u64(0, 0)
    }
}

impl Cuda {
    /// Constructs rng with the given seed.
    pub fn seed_from_u64(ordinal: usize, seed: u64) -> Self {
        Self::try_seed_from_u64(ordinal, seed).unwrap()
    }

    /// Constructs rng with the given seed.
    pub fn try_seed_from_u64(ordinal: usize, seed: u64) -> Result<Self, CudaError> {
        let rng = Arc::new(Mutex::new(StdRng::seed_from_u64(seed)));
        let dev = CudaDeviceBuilder::new(ordinal).build()?;
        let blas = Arc::new(CudaBlas::new(dev.clone())?);
        Ok(Self { rng, dev, blas })
    }
}

#[derive(Debug, Clone)]
pub struct CudaArray<S: Shape, E> {
    pub(crate) data: Arc<CudaSlice<E>>,
    pub(crate) shape: S,
    pub(crate) strides: S::Concrete,
}

impl<S: Shape, E> HasShape for CudaArray<S, E> {
    type WithShape<New: Shape> = CudaArray<New, S>;
    type Shape = S;
    fn shape(&self) -> &S {
        &self.shape
    }
}

impl<S: Shape, E: Unit> HasUnitType for CudaArray<S, E> {
    type Unit = E;
}

impl<S: Shape, E: Dtype> HasDtype for CudaArray<S, E> {
    type Dtype = E;
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl HasErr for Cuda {
    type Err = CudaError;
}

impl DeviceStorage for Cuda {
    type Storage<S: Shape, E: Unit> = CudaArray<S, E>;

    fn try_alloc_grad<S: Shape, E: Dtype>(
        &self,
        storage: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        self.try_new_like(storage, Default::default())
    }

    fn random_u64(&self) -> u64 {
        self.rng.lock().unwrap().gen()
    }
}

impl Cuda {
    #[inline]
    pub(crate) fn try_new_with<S: Shape, E: Default + Clone>(
        &self,
        shape: S,
        elem: E,
    ) -> Result<CudaArray<S, E>, CudaError> {
        let numel = shape.num_elements();
        let strides: S::Concrete = shape.strides();
        Ok(CudaArray {
            data: Arc::new(self.dev.take_async(std::vec![elem; numel])?),
            shape,
            strides,
        })
    }

    #[inline]
    pub(crate) fn try_new_like<S: Shape, E: Default + Clone>(
        &self,
        other: &CudaArray<S, E>,
        elem: E,
    ) -> Result<CudaArray<S, E>, CudaError> {
        let numel = other.shape.num_elements();
        let strides: S::Concrete = other.shape.strides();
        Ok(CudaArray {
            data: Arc::new(self.dev.take_async(std::vec![elem; numel])?),
            shape: other.shape,
            strides,
        })
    }
}
