use crate::shapes::{Shape, Unit};
use crate::tensor::cpu::{Cpu, CpuError, NdIndex};
use crate::tensor::{DeviceStorage, HasErr, Tensor};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr};
use cudarc::{
    cublas::{result::CublasError, CudaBlas},
    driver::{CudaDevice, CudaSlice, CudaStream, DeviceSlice, DriverError},
};

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, MutexGuard, RwLock},
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
    pub(crate) cache: Arc<RwLock<BTreeMap<usize, Vec<cudarc::driver::sys::CUdeviceptr>>>>,
}

#[derive(Debug)]
pub enum CudaError {
    Blas(CublasError),
    #[cfg(feature = "cudnn")]
    Cudnn(cudarc::cudnn::CudnnError),
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

#[cfg(feature = "cudnn")]
impl From<cudarc::cudnn::CudnnError> for CudaError {
    fn from(value: cudarc::cudnn::CudnnError) -> Self {
        Self::Cudnn(value)
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

    /// Constructs with the given seed & device ordinal
    pub fn try_build(ordinal: usize, seed: u64) -> Result<Self, CudaError> {
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
    pub(crate) unsafe fn alloc_empty<E: DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<E>, CudaError> {
        let num_bytes = len * std::mem::size_of::<E>();
        let reuse = {
            let cache = self.cache.read().unwrap();
            cache.contains_key(&num_bytes)
        };
        if reuse {
            let mut cache = self.cache.write().unwrap();
            let items = cache.get_mut(&num_bytes).unwrap();
            let allocation: CUdeviceptr = items.pop().unwrap();
            if items.is_empty() {
                cache.remove(&num_bytes);
            }
            Ok(self.dev.upgrade_device_ptr(allocation, len))
        } else {
            let out = self.dev.alloc::<E>(len)?;
            Ok(out)
        }
    }
    #[allow(unused)]
    pub(crate) unsafe fn get_workspace<E>(
        &self,
        len: usize,
    ) -> Result<MutexGuard<CudaSlice<u8>>, CudaError> {
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

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl HasErr for Cuda {
    type Err = CudaError;
}

#[derive(Debug)]
pub struct CachableCudaSlice<E> {
    pub(crate) data: CudaSlice<E>,
    pub(crate) destination: Arc<RwLock<BTreeMap<usize, Vec<cudarc::driver::sys::CUdeviceptr>>>>,
}

impl<E: cudarc::driver::DeviceRepr> Clone for CachableCudaSlice<E> {
    fn clone(&self) -> Self {
        let len = self.data.len();
        let num_bytes = self.data.num_bytes();
        let reuse = {
            let cache = self.destination.read().unwrap();
            cache.contains_key(&num_bytes)
        };
        let data = if reuse {
            let mut cache = self.destination.write().unwrap();
            let items = cache.get_mut(&num_bytes).unwrap();
            let allocation: CUdeviceptr = items.pop().unwrap();
            if items.is_empty() {
                cache.remove(&num_bytes);
            }
            let dev = self.data.device();
            let mut slice = unsafe { dev.upgrade_device_ptr(allocation, len) };
            dev.dtod_copy(&self.data, &mut slice).unwrap();
            slice
        } else {
            self.data.try_clone().unwrap()
        };
        Self {
            data,
            destination: self.destination.clone(),
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
        let dev = self.data.device();
        let null = dev.null().unwrap();
        let data = std::mem::replace(&mut self.data, null);
        let num_bytes = data.num_bytes();
        let ptr = data.leak();

        let mut cache = self.destination.write().unwrap();
        if let std::collections::btree_map::Entry::Vacant(e) = cache.entry(num_bytes) {
            e.insert(std::vec![ptr]);
        } else {
            cache.get_mut(&num_bytes).unwrap().push(ptr);
        }
    }
}

impl DeviceStorage for Cuda {
    type Vec<E: Unit> = CachableCudaSlice<E>;

    fn try_alloc_len<E: Unit>(&self, len: usize) -> Result<Self::Vec<E>, Self::Err> {
        let mut data = unsafe { self.alloc_empty(len) }?;
        self.dev.memset_zeros(&mut data)?;
        Ok(CachableCudaSlice {
            data,
            destination: self.cache.clone(),
        })
    }

    fn random_u64(&self) -> u64 {
        self.cpu.random_u64()
    }

    fn len<E: Unit>(&self, v: &Self::Vec<E>) -> usize {
        v.len()
    }

    fn tensor_to_vec<S: Shape, E: Unit, T>(&self, tensor: &Tensor<S, E, Self, T>) -> Vec<E> {
        let buf: Vec<E> = tensor.data.data.try_clone().unwrap().try_into().unwrap();
        debug_assert_eq!(buf.len(), tensor.data.len());
        let mut idx = NdIndex::new(tensor.shape, tensor.strides);
        let mut contiguous = Vec::with_capacity(tensor.shape.num_elements());
        while let Some(i) = idx.next() {
            contiguous.push(buf[i]);
        }
        contiguous
    }

    fn try_synchronize(&self) -> Result<(), CudaError> {
        self.dev.synchronize().map_err(CudaError::from)
    }
}
