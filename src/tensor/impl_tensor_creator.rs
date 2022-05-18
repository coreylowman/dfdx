use super::*;
use crate::prelude::*;
use rand::prelude::Distribution;

pub trait TensorCreator: HasArrayData + Sized {
    fn new_boxed(data: Box<Self::Array>) -> Self;

    fn new(data: Self::Array) -> Self {
        Self::new_boxed(Box::new(data))
    }

    fn zeros() -> Self {
        Self::new_boxed(Cpu::zeros())
    }

    fn ones() -> Self
    where
        Cpu: FillElements<Self::Array>,
    {
        Self::new_boxed(Cpu::filled(&mut |f| *f = 1.0))
    }

    fn rand<R: rand::Rng>(rng: &mut R) -> Self
    where
        Cpu: FillElements<Self::Array>,
    {
        Self::new_boxed(Cpu::filled(&mut |f| *f = rand_distr::Standard.sample(rng)))
    }

    fn randn<R: rand::Rng>(rng: &mut R) -> Self
    where
        Cpu: FillElements<Self::Array>,
    {
        Self::new_boxed(Cpu::filled(&mut |f| {
            *f = rand_distr::StandardNormal.sample(rng)
        }))
    }
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> TensorCreator for $typename<$($Vs, )* NoTape> {
    fn new_boxed(data: Box<Self::Array>) -> Self {
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
        assert_eq!(Tensor2D::new(t.clone()).data(), &t);
    }

    #[test]
    fn fuzz_test_rand() {
        let mut rng = thread_rng();
        for &v in Tensor1D::<1000>::rand(&mut rng).data() {
            assert!(0.0 <= v && v < 1.0);
        }
    }

    #[test]
    fn test_randn() {
        let mut rng = thread_rng();
        let _t = Tensor1D::<1000>::randn(&mut rng);
    }
}
