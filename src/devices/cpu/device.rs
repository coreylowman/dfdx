use crate::arrays::{Dtype, HasShape, Shape, StridesFor};
use crate::devices::device::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{cell::RefCell, sync::Arc, vec::Vec};

#[derive(Clone, Debug)]
pub struct Cpu {
    pub(crate) rng: Arc<RefCell<StdRng>>,
}

impl Default for Cpu {
    fn default() -> Self {
        Self {
            rng: Arc::new(RefCell::new(StdRng::seed_from_u64(0))),
        }
    }
}

impl Cpu {
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: Arc::new(RefCell::new(StdRng::seed_from_u64(seed))),
        }
    }
}

#[derive(Debug)]
pub struct StridedArray<S: Shape, Elem> {
    pub(crate) data: Arc<Vec<Elem>>,
    pub(crate) shape: S,
    pub(crate) strides: StridesFor<S>,
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
    fn alloc<S: Shape, E: Dtype>(&self, shape: &S) -> Result<Self::Storage<S, E>, Self::Err> {
        self.try_zeros_like(*shape)
    }
    fn alloc_like<S: Shape, E: Dtype>(
        &self,
        storage: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        self.try_zeros_like(storage)
    }
    fn random_u64(&self) -> u64 {
        self.rng.borrow_mut().gen()
    }

    fn fill_with<S: Shape, E: Dtype>(&self, storage: &mut Self::Storage<S, E>, value: E) {
        for v in storage.buf_iter_mut() {
            *v = value;
        }
    }
}
