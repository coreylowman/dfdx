use crate::shapes::{Dtype, HasDtype, HasShape, HasUnitType, Shape, Unit};
use crate::tensor::storage_traits::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    sync::{Arc, Mutex},
    vec::Vec,
};

/// A device that stores data on the heap.
///
/// The [Default] impl seeds the underlying rng with seed of 0.
///
/// Use [Cpu::seed_from_u64] to control what seed is used.
#[derive(Clone, Debug)]
pub struct Cpu {
    pub(crate) rng: Arc<Mutex<StdRng>>,
}

impl Default for Cpu {
    fn default() -> Self {
        Self {
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(0))),
        }
    }
}

impl Cpu {
    /// Constructs rng with the given seed.
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
        }
    }
}

/// The storage for the cpu device
#[derive(Debug, Clone)]
pub struct StridedArray<S: Shape, E> {
    pub(crate) data: Arc<Vec<E>>,
    pub(crate) shape: S,
    pub(crate) strides: S::Concrete,
}

#[derive(Debug, Clone, Copy)]
pub enum CpuError {
    /// Device is out of memory
    OutOfMemory,
    /// Not enough elements were provided when creating a tensor
    NotEnoughElements,
}

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfMemory => f.write_str("CpuError::OutOfMemory"),
            Self::NotEnoughElements => f.write_str("CpuError::NotEnoughElements"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CpuError {}

impl<S: Shape, E> HasShape for StridedArray<S, E> {
    type WithShape<New: Shape> = StridedArray<New, S>;
    type Shape = S;
    fn shape(&self) -> &S {
        &self.shape
    }
}

impl<S: Shape, E: Unit> HasUnitType for StridedArray<S, E> {
    type Unit = E;
}

impl<S: Shape, E: Dtype> HasDtype for StridedArray<S, E> {
    type Dtype = E;
}

impl HasErr for Cpu {
    type Err = CpuError;
}

impl DeviceStorage for Cpu {
    type Storage<S: Shape, E: Unit> = StridedArray<S, E>;

    fn try_alloc_grad<S: Shape, E: Dtype>(
        &self,
        storage: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        StridedArray::try_new_like(storage, Default::default())
    }

    fn random_u64(&self) -> u64 {
        self.rng.lock().unwrap().gen()
    }
}
