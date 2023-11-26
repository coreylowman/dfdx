mod allocate;
mod device;

pub use device::Buffer;
pub use device::Webgpu;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shapes::*, tensor::*};

    #[test]
    fn test_empty_cache() {
        let dev: Webgpu = Default::default();
        dev.enable_cache();
        let tensor: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        drop(tensor); // insert allocation into cache
        assert_eq!(dev.cache.len(), 1);
        dev.empty_cache();
        assert_eq!(dev.cache.len(), 0);
    }

    #[test]
    fn test_disabling_cache_empties_it() {
        let dev: Webgpu = Default::default();
        dev.enable_cache();
        let tensor: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        drop(tensor); // insert allocation into cache
        assert_eq!(dev.cache.len(), 1);
        dev.disable_cache();
        assert_eq!(dev.cache.len(), 0);
    }

    #[test]
    fn test_reuse_allocation_on_new_tensor() {
        let dev: Webgpu = Default::default();
        dev.enable_cache();
        let tensor: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        let id = tensor.data.data.global_id();
        drop(tensor); // insert allocation into cache
        assert_eq!(dev.cache.len(), 1);
        let other: Tensor<Rank2<2, 3>, f64, _> = dev.zeros();
        assert_eq!(dev.cache.len(), 1);
        let tensor: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        assert_eq!(dev.cache.len(), 0);
        assert_eq!(tensor.data.data.global_id(), id);
        drop(other);
    }

    #[test]
    fn test_reuse_allocation_on_clone_tensor() {
        let dev: Webgpu = Default::default();
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

    #[test]
    fn test_ones_like() {
        let dev: Webgpu = Default::default();
        let a: Tensor<Rank2<2, 3>, f32, _> = dev.ones();
        let b: Tensor<Rank2<2, 3>, f32, _> = dev.ones_like(&a);
        assert_eq!(a.array(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
        assert_eq!(a.array(), b.array());
    }
}
