use super::*;
use ndarray::Array;
use rand::prelude::Distribution;

pub trait TensorCreator: Tensor + Sized {
    fn new(data: Array<f32, Self::Dimension>) -> Self;

    fn zeros() -> Self {
        Self::new(Array::zeros(Self::SHAPE))
    }

    fn ones() -> Self {
        Self::new(Array::ones(Self::SHAPE))
    }

    fn rand<R: rand::Rng>(rng: &mut R) -> Self {
        Self::new(Array::from_shape_simple_fn(Self::SHAPE, || {
            rand_distr::Standard.sample(rng)
        }))
    }

    fn randn<R: rand::Rng>(rng: &mut R) -> Self {
        Self::new(Array::from_shape_simple_fn(Self::SHAPE, || {
            rand_distr::StandardNormal.sample(rng)
        }))
    }
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> TensorCreator for $typename<$($Vs, )* NoTape> {
    fn new(data: Array<f32, Self::Dimension>) -> Self {
        assert!(data.shape() == Self::SHAPE_SLICE);
        Self {
            id: unique_id(),
            data,
            tape: NoTape::default(),
        }
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);

fn unique_id() -> usize {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
