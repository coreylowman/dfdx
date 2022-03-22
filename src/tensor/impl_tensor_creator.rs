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

fn unique_id() -> usize {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl TensorCreator for Tensor0D<NoTape> {
    fn new(data: Array<f32, Self::Dimension>) -> Self {
        assert!(data.dim() == Self::SHAPE);
        Self {
            id: unique_id(),
            data,
            tape: NoTape::default(),
        }
    }
}

impl<const N: usize> TensorCreator for Tensor1D<N, NoTape> {
    fn new(data: Array<f32, Self::Dimension>) -> Self {
        assert!(data.dim() == Self::SHAPE.0);
        Self {
            id: unique_id(),
            data,
            tape: NoTape::default(),
        }
    }
}

impl<const M: usize, const N: usize> TensorCreator for Tensor2D<M, N, NoTape> {
    fn new(data: Array<f32, Self::Dimension>) -> Self {
        assert!(data.dim() == Self::SHAPE);
        Self {
            id: unique_id(),
            data,
            tape: NoTape::default(),
        }
    }
}

impl<const M: usize, const N: usize, const O: usize> TensorCreator for Tensor3D<M, N, O, NoTape> {
    fn new(data: Array<f32, Self::Dimension>) -> Self {
        assert!(data.dim() == Self::SHAPE);
        Self {
            id: unique_id(),
            data,
            tape: NoTape::default(),
        }
    }
}
impl<const M: usize, const N: usize, const O: usize, const P: usize> TensorCreator
    for Tensor4D<M, N, O, P, NoTape>
{
    fn new(data: Array<f32, Self::Dimension>) -> Self {
        assert!(data.dim() == Self::SHAPE);
        Self {
            id: unique_id(),
            data,
            tape: NoTape::default(),
        }
    }
}
