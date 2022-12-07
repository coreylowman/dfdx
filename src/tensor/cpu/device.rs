use crate::shapes::{Dtype, HasDtype, HasShape, Shape};
use crate::tensor::storage_traits::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    sync::{Arc, Mutex},
    vec::Vec,
};

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
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StridedArray<S: Shape, Elem> {
    pub(crate) data: Arc<Vec<Elem>>,
    pub(crate) shape: S,
    pub(crate) strides: S::Concrete,
}

#[derive(Debug, Clone, Copy)]
pub enum CpuError {
    OutOfMemory,
}

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfMemory => f.write_str("CpuError::OutOfMemory"),
        }
    }
}

impl<S: Shape, E> HasShape for StridedArray<S, E> {
    type WithShape<New: Shape> = StridedArray<New, S>;
    type Shape = S;
    fn shape(&self) -> &S {
        &self.shape
    }
}

impl<S: Shape, E: Dtype> HasDtype for StridedArray<S, E> {
    type Dtype = E;
}

impl HasErr for Cpu {
    type Err = CpuError;
}

impl DeviceStorage for Cpu {
    type Storage<S: Shape, E: Dtype> = StridedArray<S, E>;

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
