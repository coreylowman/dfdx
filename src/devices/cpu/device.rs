use super::iterate::LendingIterator;
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

impl<const N: usize, S: Shape<Concrete = [usize; N]>, E: Dtype + std::ops::SubAssign<E>>
    std::ops::SubAssign for StridedArray<S, E>
{
    fn sub_assign(&mut self, rhs: Self) {
        let mut lhs_iter = self.iter_mut();
        let mut rhs_iter = rhs.iter();
        while let Some((l, r)) = lhs_iter.next().zip(rhs_iter.next()) {
            *l -= *r;
        }
    }
}

impl Device for Cpu {
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
}
