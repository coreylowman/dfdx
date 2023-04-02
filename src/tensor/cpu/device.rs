use crate::shapes::{Shape, Unit};
use crate::tensor::{storage_traits::*, Tensor};
use core::marker::PhantomData;
use rand::Rng;
use rand::{rngs::StdRng, SeedableRng};
use std::{sync::Arc, vec::Vec};

#[cfg(feature = "no-std")]
use spin::Mutex;

#[cfg(not(feature = "no-std"))]
use std::sync::Mutex;

/// A device that stores data on the heap.
///
/// The [Default] impl seeds the underlying rng with seed of 0.
///
/// Use [Cpu::seed_from_u64] to control what seed is used.
#[derive(Clone, Debug)]
pub struct Cpu<S = VecStorage> {
    pub(crate) rng: Arc<Mutex<StdRng>>,
    _storage: PhantomData<S>,
}

impl<E: Unit, S: DeviceStorage<E>> DeviceStorage<E> for Cpu<S> {
    type Storage = S::Storage;
}

impl<S> RandomU64 for Cpu<S> {
    fn random_u64(&self) -> u64 {
        #[cfg(not(feature = "no-std"))]
        {
            self.rng.lock().unwrap().gen()
        }
        #[cfg(feature = "no-std")]
        {
            self.rng.lock().gen()
        }
    }
}

impl<S> Default for Cpu<S> {
    fn default() -> Self {
        Self {
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(0))),
            _storage: PhantomData,
        }
    }
}

impl<S> Cpu<S> {
    /// Constructs rng with the given seed.
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
            _storage: PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CpuError {
    /// Device is out of memory
    OutOfMemory,
    /// Not enough elements were provided when creating a tensor
    WrongNumElements,
}

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfMemory => f.write_str("CpuError::OutOfMemory"),
            Self::WrongNumElements => f.write_str("CpuError::WrongNumElements"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CpuError {}

impl<G> HasErr for Cpu<G> {
    type Err = CpuError;
}

impl<E: Unit, G: 'static + DeviceStorage<E>> DeviceAllocGrad<E> for Cpu<G> {
    fn try_alloc_grad(&self, other: &G::Storage) -> Result<G::Storage, Self::Err> {
        self.try_alloc_zeros(other.len())
    }
}

impl<E: Unit, G: DeviceStorage<E>> DeviceTensorToVec<E> for Cpu<G> {
    fn tensor_to_vec<S: Shape, T>(&self, tensor: &Tensor<S, E, Self, T>) -> Vec<E> {
        let mut buf = Vec::with_capacity(tensor.shape.num_elements());
        for v in tensor.iter_copied() {
            buf.push(v)
        }
        buf
    }
}
