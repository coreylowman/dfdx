use super::*;
use crate::array_ops::{FillElements, ZeroElements};
use rand::prelude::Distribution;

pub trait TensorCreator: Tensor + Sized {
    fn new(data: Self::ArrayType) -> Self;

    fn zeros() -> Self {
        Self::new(Self::ArrayType::ZEROS)
    }

    fn ones() -> Self {
        Self::new(Self::ArrayType::filled_with(&mut || 1.0))
    }

    fn rand<R: rand::Rng>(rng: &mut R) -> Self {
        Self::new(Self::ArrayType::filled_with(&mut || {
            rand_distr::Standard.sample(rng)
        }))
    }

    fn randn<R: rand::Rng>(rng: &mut R) -> Self {
        Self::new(Self::ArrayType::filled_with(&mut || {
            rand_distr::StandardNormal.sample(rng)
        }))
    }
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> TensorCreator for $typename<$($Vs, )* NoTape> {
    fn new(data: Self::ArrayType) -> Self {
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
