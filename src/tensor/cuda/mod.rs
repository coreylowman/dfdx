mod allocate;
mod device;

pub use device::{Cuda, CudaError};

pub(crate) fn launch_cfg<const NUM_THREADS: u32>(n: u32) -> cudarc::driver::LaunchConfig {
    let num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
    cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shapes::*, tensor::*};
    use cudarc::driver::DevicePtr;

    #[test]
    fn test_empty_cache() {
        let dev: Cuda = Default::default();
        dev.enable_cache();
        let tensor: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        drop(tensor); // insert allocation into cache
        assert_eq!(dev.cache.len(), 1);
        dev.empty_cache();
        assert_eq!(dev.cache.len(), 0);
    }

    #[test]
    fn test_disabling_cache_empties_it() {
        let dev: Cuda = Default::default();
        dev.enable_cache();
        let tensor: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        drop(tensor); // insert allocation into cache
        assert_eq!(dev.cache.len(), 1);
        dev.disable_cache();
        assert_eq!(dev.cache.len(), 0);
    }

    #[test]
    fn test_reuse_allocation_on_new_tensor() {
        let dev: Cuda = Default::default();
        dev.enable_cache();
        let tensor: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        let ptr = *tensor.data.device_ptr();
        drop(tensor); // insert allocation into cache
        assert_eq!(dev.cache.len(), 1);
        let other: Tensor<Rank2<2, 3>, f64, _> = dev.zeros();
        assert_eq!(dev.cache.len(), 1);
        let tensor: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        assert_eq!(dev.cache.len(), 0);
        assert_eq!(*tensor.data.device_ptr(), ptr);
        drop(other);
    }

    #[test]
    fn test_reuse_allocation_on_clone_tensor() {
        let dev: Cuda = Default::default();
        dev.enable_cache();
        let a: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        let b: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        drop(b); // insert allocation into cache
        assert_eq!(dev.cache.len(), 1);
        let mut b = a.clone();
        assert_eq!(dev.cache.len(), 1);
        // will actually clone the data - should reuse allocation from cache
        std::sync::Arc::make_mut(&mut b.data);
        assert_eq!(dev.cache.len(), 0);
    }
}
