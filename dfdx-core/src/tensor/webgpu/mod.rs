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
    fn test_new_allocation_on_clone_tensor() {
        let dev: Webgpu = Default::default();
        dev.enable_cache();
        let a: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        let b: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        let mut b = a.clone();
        assert_eq!(dev.cache.len(), 0);
        dev.synchronize();
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

    #[test]
    fn test_copy() {
        let dev: Webgpu = Default::default();
        let mut b: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        b.copy_from(&[1.0; 6]);
        assert_eq!(b.array(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
        let mut slice = [0.0; 6];
        b.copy_into(&mut slice);
        assert_eq!(slice, [1.0; 6]);
    }

    #[test]
    fn test_fill_zeros() {
        let dev: Webgpu = Default::default();
        let mut b: Tensor<Rank2<2, 3>, f32, _> = dev.ones();
        assert_eq!(b.array(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
        b.fill_with_zeros();
        assert_eq!(b.array(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    }

    #[test]
    fn test_fill_ones() {
        let dev: Webgpu = Default::default();
        let mut b: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
        assert_eq!(b.array(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        b.fill_with_ones();
        assert_eq!(b.array(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    }

    #[test]
    fn test_sample() {
        let dev: Webgpu = Default::default();
        let b: Tensor<Rank2<2, 3>, f32, _> = dev.sample_uniform();
        assert_eq!(
            b.array(),
            [
                [0.80145925, 0.7311134, 0.55528885],
                [0.77346015, 0.809342, 0.025844634]
            ]
        );
    }

    #[test]
    fn test_from_vec() {
        let dev: Webgpu = Default::default();
        let b: Tensor<Rank2<2, 3>, f32, _> =
            dev.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (Const::<2>, Const::<3>));
        assert_eq!(b.array(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    }
}
