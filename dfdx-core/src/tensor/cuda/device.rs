use crate::shapes::{Shape, Unit};
use crate::tensor::cpu::Cpu;
use crate::tensor::{
    cache::TensorCache, Cache, Error, NoneTape, RandomU64, Storage, Synchronize, Tensor,
};

use cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr};
use cudarc::{
    cublas::{result::CublasError, CudaBlas},
    driver::sys::CUdeviceptr,
    driver::{CudaDevice, CudaSlice, CudaStream, DeviceSlice, DriverError},
};

use std::{
    sync::{Arc, Mutex, MutexGuard},
    vec::Vec,
};

/// A Cuda device that enables constructing tensors on GPUs
/// & running GPU kernels.
#[derive(Clone, Debug)]
pub struct Cuda {
    pub(crate) cpu: Cpu,
    pub(crate) dev: Arc<CudaDevice>,
    pub(crate) blas: Arc<CudaBlas>,
    #[cfg(feature = "cudnn")]
    #[allow(unused)]
    pub(crate) cudnn: Arc<cudarc::cudnn::Cudnn>,
    /// A second stream for kernels to optionally execute on.
    pub(crate) par_stream: Arc<CudaStream>,
    pub(crate) workspace: Arc<Mutex<CudaSlice<u8>>>,
    pub(crate) cache: Arc<TensorCache<CUdeviceptr>>,
}

impl From<CublasError> for Error {
    fn from(value: CublasError) -> Self {
        Self::CublasError(value)
    }
}

impl From<DriverError> for Error {
    fn from(value: DriverError) -> Self {
        Self::CudaDriverError(value)
    }
}

#[cfg(feature = "cudnn")]
impl From<cudarc::cudnn::CudnnError> for Error {
    fn from(value: cudarc::cudnn::CudnnError) -> Self {
        Self::CudnnError(value)
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
    pub fn try_seed_from_u64(seed: u64) -> Result<Self, Error> {
        Self::try_build(0, seed)
    }

    /// Constructs with the given seed & device ordinal
    pub fn try_build(ordinal: usize, seed: u64) -> Result<Self, Error> {
        let cpu = Cpu::seed_from_u64(seed);
        let dev = CudaDevice::new(ordinal)?;
        let blas = Arc::new(CudaBlas::new(dev.clone())?);
        #[cfg(feature = "cudnn")]
        let cudnn = cudarc::cudnn::Cudnn::new(dev.clone())?;
        let par_stream = Arc::new(dev.fork_default_stream()?);
        let workspace = Arc::new(Mutex::new(dev.alloc_zeros::<u8>(0)?));
        Ok(Self {
            cpu,
            dev,
            blas,
            #[cfg(feature = "cudnn")]
            cudnn,
            par_stream,
            workspace,
            cache: Default::default(),
        })
    }
}

impl Cuda {
    /// Allocates an empty [CudaSlice] either from the cache or by allocating new memory.
    /// In either case, the memory will have uninitialized values, meaning the user must
    /// initialize it before using it.
    pub(crate) unsafe fn alloc_empty<E: DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<E>, Error> {
        let data = self.cache.try_pop::<E>(len).map_or_else(
            || self.dev.alloc::<E>(len),
            |ptr| Ok(self.dev.upgrade_device_ptr(ptr, len)),
        )?;
        Ok(data)
    }
    #[allow(unused)]
    pub(crate) unsafe fn get_workspace<E>(
        &self,
        len: usize,
    ) -> Result<MutexGuard<CudaSlice<u8>>, Error> {
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

/// A [CudaSlice] that can be cloned without allocating new memory.
/// When [Drop]ed it will insert it's data into the cache.
#[derive(Debug)]
pub struct CachableCudaSlice<E> {
    /// The actual data.
    pub(crate) data: CudaSlice<E>,
    /// A cache of device pointers that can be reused.
    pub(crate) cache: Arc<TensorCache<CUdeviceptr>>,
}

impl<E: cudarc::driver::DeviceRepr> Clone for CachableCudaSlice<E> {
    fn clone(&self) -> Self {
        let dev = self.data.device();
        let len = self.data.len();
        let data = self.cache.try_pop::<E>(len).map_or_else(
            || self.data.try_clone().unwrap(),
            |ptr| {
                // SAFETY:
                // 1. we know that ptr is valid for `num_bytes` because it was registered for that.
                // 2. we are about to set the memory with dtod_copy
                let mut slice = unsafe { dev.upgrade_device_ptr(ptr, len) };
                dev.dtod_copy(&self.data, &mut slice).unwrap();
                slice
            },
        );
        Self {
            data,
            cache: self.cache.clone(),
        }
    }
}

unsafe impl<E: DeviceRepr> DeviceRepr for &CachableCudaSlice<E> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self.data.device_ptr() as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

unsafe impl<E: DeviceRepr> DeviceRepr for &mut CachableCudaSlice<E> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self.data.device_ptr() as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

impl<E> DeviceSlice<E> for CachableCudaSlice<E> {
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<E> DevicePtr<E> for CachableCudaSlice<E> {
    fn device_ptr(&self) -> &cudarc::driver::sys::CUdeviceptr {
        self.data.device_ptr()
    }
}

impl<E> DevicePtrMut<E> for CachableCudaSlice<E> {
    fn device_ptr_mut(&mut self) -> &mut cudarc::driver::sys::CUdeviceptr {
        self.data.device_ptr_mut()
    }
}

impl<E> std::ops::Deref for CachableCudaSlice<E> {
    type Target = CudaSlice<E>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<E> std::ops::DerefMut for CachableCudaSlice<E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<E> Drop for CachableCudaSlice<E> {
    fn drop(&mut self) {
        if self.cache.is_enabled() {
            let dev = self.data.device();
            // Replaces the CudaSlice with a 0 length CudaSlice. This won't take additional
            // memory, but will give us ownership of the actual data.
            let data = std::mem::replace(&mut self.data, dev.null().unwrap());
            let numel = data.len();
            // Get access to the raw pointer without freeing it.
            let ptr = data.leak();
            self.cache.insert::<E>(numel, ptr);
        }
    }
}

impl RandomU64 for Cuda {
    fn random_u64(&self) -> u64 {
        self.cpu.random_u64()
    }
}

impl Cache for Cuda {
    fn try_enable_cache(&self) -> Result<(), Error> {
        self.cache.enable();
        Ok(())
    }

    fn try_disable_cache(&self) -> Result<(), Error> {
        self.cache.disable();
        self.try_empty_cache()
    }

    fn try_empty_cache(&self) -> Result<(), Error> {
        #[cfg(not(feature = "no-std"))]
        let mut cache = self.cache.allocations.write().unwrap();
        #[cfg(feature = "no-std")]
        let mut cache = self.cache.allocations.write();
        for (&key, allocations) in cache.iter_mut() {
            for alloc in allocations.drain(..) {
                let data = unsafe { self.dev.upgrade_device_ptr::<u8>(alloc, key.num_bytes) };
                drop(data);
            }
        }
        cache.clear();
        Ok(())
    }
}

impl Synchronize for Cuda {
    fn try_synchronize(&self) -> Result<(), Error> {
        self.dev.synchronize().map_err(Error::from)
    }
}

impl<E: Unit> Storage<E> for Cuda {
    type Vec = CachableCudaSlice<E>;

    fn try_alloc_len(&self, len: usize) -> Result<Self::Vec, Error> {
        let mut data = unsafe { self.alloc_empty(len) }?;
        self.dev.memset_zeros(&mut data)?;
        Ok(CachableCudaSlice {
            data,
            cache: self.cache.clone(),
        })
    }

    fn len(&self, v: &Self::Vec) -> usize {
        v.len()
    }

    fn tensor_to_vec<S: Shape, T>(&self, tensor: &Tensor<S, E, Self, T>) -> Vec<E> {
        let buf = self
            .cpu
            .try_alloc_elem(tensor.data.data.len(), Default::default())
            .unwrap();
        let mut cpu_tensor = Tensor {
            id: tensor.id,
            data: Arc::new(buf),
            shape: tensor.shape,
            strides: tensor.strides,
            device: self.cpu.clone(),
            tape: NoneTape,
        };
        let buf = std::sync::Arc::get_mut(&mut cpu_tensor.data).unwrap();
        self.dev
            .dtoh_sync_copy_into(&tensor.data.data, &mut buf.data)
            .unwrap();
        self.cpu.tensor_to_vec::<S, _>(&cpu_tensor)
    }
}
