use crate::arrays::{Dtype, HasShape, Shape, StridesFor};
use crate::devices::device::*;
use rand::{rngs::StdRng, SeedableRng};
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

#[derive(Debug)]
pub struct StridedArray<S: Shape, Elem> {
    pub(super) data: Arc<Vec<Elem>>,
    pub(super) shape: S,
    pub(super) strides: StridesFor<S>,
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

impl<S: Shape, E: Clone> StridedArray<S, E> {
    pub(super) fn try_clone(&self) -> Result<Self, CpuError> {
        // TODO fallible version cloning vec
        Ok(StridedArray {
            data: self.data.clone(),
            shape: self.shape,
            strides: self.strides,
        })
    }

    pub(super) fn strided_ptr(&self) -> (*const E, &S::Concrete) {
        (self.data.as_ptr(), &self.strides.0)
    }

    pub(super) fn strided_ptr_mut(&mut self) -> (*mut E, &S::Concrete) {
        let data = Arc::make_mut(&mut self.data);
        (data.as_mut_ptr(), &self.strides.0)
    }
}

impl<S: Shape, E> HasShape for StridedArray<S, E> {
    type Shape = S;
    fn shape(&self) -> &S {
        &self.shape
    }
}

impl Device for Cpu {
    type Storage<S: Shape, E: Dtype> = StridedArray<S, E>;
    type Err = CpuError;
    fn alloc<S: Shape, E: Dtype>(&self, shape: &S) -> Result<Self::Storage<S, E>, Self::Err> {
        todo!();
    }
    fn sub_assign<S: Shape, E: Dtype>(
        &self,
        lhs: &mut Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
    ) {
        todo!();
    }
}
