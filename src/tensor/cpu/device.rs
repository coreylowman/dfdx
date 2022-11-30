use crate::arrays::{Dtype, HasShape, Shape};
use crate::tensor::storage::*;
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

#[derive(Debug)]
pub struct StridedArray<S: Shape, Elem> {
    pub(crate) data: Arc<Vec<Elem>>,
    pub(crate) shape: S,
    pub(crate) strides: S::Concrete,
}

impl<S: Shape, E: Clone> Clone for StridedArray<S, E> {
    fn clone(&self) -> Self {
        self.try_clone().unwrap()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CpuError {
    OutOfMemory,
    ShapeMismatch,
}

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        todo!();
    }
}

impl<S: Shape, E: Clone> StridedArray<S, E> {
    pub(crate) fn try_clone(&self) -> Result<Self, CpuError> {
        // TODO fallible version cloning vec
        Ok(StridedArray {
            data: self.data.clone(),
            shape: self.shape,
            strides: self.strides,
        })
    }
}

impl<S: Shape, E> HasShape for StridedArray<S, E> {
    type Shape = S;
    fn shape(&self) -> &S {
        &self.shape
    }
}

impl HasErr for Cpu {
    type Err = CpuError;
}

impl DeviceStorage for Cpu {
    type Storage<S: Shape, E: Dtype> = StridedArray<S, E>;
    fn try_alloc<S: Shape, E: Dtype>(&self, shape: &S) -> Result<Self::Storage<S, E>, Self::Err> {
        StridedArray::try_new_with(*shape, Default::default())
    }

    fn try_alloc_like<S: Shape, E: Dtype>(
        &self,
        storage: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        StridedArray::try_new_like(storage, Default::default())
    }

    fn random_u64(&self) -> u64 {
        self.rng.lock().unwrap().gen()
    }
}
