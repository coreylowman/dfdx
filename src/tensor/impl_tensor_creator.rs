use super::*;
use crate::prelude::*;
use num_traits::One;
use rand::prelude::Distribution;
use rand_distr::{Standard, StandardNormal};

/// Something that can be created - currently only implemented for tensors with no tapes.
pub trait TensorCreator: Sized + HasDevice {
    /// Create a new tensor with a `Box<Self::Array>`.
    fn new_boxed(data: Box<Self::Array>) -> Self;

    /// Create a new tensor with `Self::Array` on the stack. This just boxes `Self::Array` and calls [TensorCreator::new_boxed].
    fn new(data: Self::Array) -> Self {
        Self::new_boxed(Box::new(data))
    }

    /// Creates a tensor filled with all 0s.
    fn zeros() -> Self {
        Self::new_boxed(Self::Device::zeros())
    }

    /// Creates a tensor filled with all 1s.
    fn ones() -> Self
    where
        Self::Dtype: One,
    {
        Self::new_boxed(Self::Device::filled(&mut |v| *v = One::one()))
    }

    /// Creates a tensor filled with values sampled from [Standard] distribution.
    fn rand<R: rand::Rng>(rng: &mut R) -> Self
    where
        Standard: Distribution<Self::Dtype>,
    {
        Self::new_boxed(Self::Device::filled(&mut |v| *v = Standard.sample(rng)))
    }

    /// Creates a tensor filled with values sampled from [StandardNormal] distribution.
    fn randn<R: rand::Rng>(rng: &mut R) -> Self
    where
        StandardNormal: Distribution<Self::Dtype>,
    {
        Self::new_boxed(Self::Device::filled(&mut |v| {
            *v = StandardNormal.sample(rng)
        }))
    }
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> TensorCreator for $typename<$($Vs, )* NoneTape> {
    /// Returns a new object with `data` and a new [UniqueId].
    fn new_boxed(data: Box<Self::Array>) -> Self {
        Self {
            id: unique_id(),
            data,
            tape: Default::default(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unique_id::{unique_id, UniqueId};
    use rand::thread_rng;

    #[test]
    fn test_id() {
        let base = unique_id().0;
        assert_eq!(Tensor0D::new(0.0).id, UniqueId(base + 1));
        assert_eq!(Tensor0D::new(0.0).id, UniqueId(base + 2));
        assert_eq!(Tensor1D::<5>::zeros().id, UniqueId(base + 3));
        assert_eq!(Tensor2D::<3, 2>::ones().id, UniqueId(base + 4));
        assert_eq!(Tensor3D::<4, 2, 3>::zeros().id, UniqueId(base + 5));
    }

    #[test]
    fn test_zeros() {
        assert_eq!(Tensor2D::<3, 2>::zeros().data(), &[[0.0; 2]; 3]);
    }

    #[test]
    fn test_ones() {
        assert_eq!(Tensor2D::<3, 2>::ones().data(), &[[1.0; 2]; 3]);
    }

    #[test]
    fn test_new() {
        let t = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(Tensor2D::new(t).data(), &t);
    }

    #[test]
    fn fuzz_test_rand() {
        let mut rng = thread_rng();
        for &v in Tensor1D::<1000>::rand(&mut rng).data() {
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_randn() {
        let mut rng = thread_rng();
        let _t = Tensor1D::<1000>::randn(&mut rng);
    }
}
