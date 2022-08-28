use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

pub trait ConstPermute2<const I: isize, const J: isize> {
    type Permuted;
    fn const_permute(self) -> Self::Permuted;
}

pub trait ConstPermute3<const I: isize, const J: isize, const K: isize> {
    type Permuted;
    fn const_permute(self) -> Self::Permuted;
}

pub trait ConstPermute4<const I: isize, const J: isize, const K: isize, const L: isize> {
    type Permuted;
    fn const_permute(self) -> Self::Permuted;
}

#[rustfmt::skip]
macro_rules! axis { (0) => { M }; (1) => { N }; (2) => { O }; (3) => { P }; }

#[rustfmt::skip]
macro_rules! tensor {
    ($Ax0:tt) => { Tensor1D<axis!($Ax0), H> };
    ($Ax0:tt, $Ax1:tt) => { Tensor2D<axis!($Ax0), axis!($Ax1), H> };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt) => { Tensor3D<axis!($Ax0), axis!($Ax1), axis!($Ax2), H> };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt) => { Tensor4D<axis!($Ax0), axis!($Ax1), axis!($Ax2), axis!($Ax3), H> };
}

#[rustfmt::skip]
macro_rules! impl_permute {
    ($Ax0:tt, $Ax1:tt) => {
impl<const M: usize, const N: usize, H: Tape> ConstPermute2<$Ax0, $Ax1> for tensor!(0, 1) {
    type Permuted = tensor!($Ax0, $Ax1);
    fn const_permute(self) -> Self::Permuted {
        let mut result: <Self::Permuted as Tensor>::NoTape = TensorCreator::zeros();
        <Cpu as DevicePermute2<_, _, $Ax0, $Ax1>>::permute(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            <Cpu as DevicePermute2<_, _, $Ax0, $Ax1>>::inverse_permute(t.mut_data(), result_grad);
            Cpu::add(t_grad, t.data());
        })
    }
}
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt) => {
impl<const M: usize, const N: usize, const O: usize, H: Tape> ConstPermute3<$Ax0, $Ax1, $Ax2> for tensor!(0, 1, 2) {
    type Permuted = tensor!($Ax0, $Ax1, $Ax2);
    fn const_permute(self) -> Self::Permuted {
        let mut result: <Self::Permuted as Tensor>::NoTape = TensorCreator::zeros();
        <Cpu as DevicePermute3<_, _, $Ax0, $Ax1, $Ax2>>::permute(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            <Cpu as DevicePermute3<_, _, $Ax0, $Ax1, $Ax2>>::inverse_permute(t.mut_data(), result_grad);
            Cpu::add(t_grad, t.data());
        })
    }
}
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt) => {
impl<const M: usize, const N: usize, const O: usize, const P: usize, H: Tape>
    ConstPermute4<$Ax0, $Ax1, $Ax2, $Ax3> for tensor!(0, 1, 2, 3)
{
    type Permuted = tensor!($Ax0, $Ax1, $Ax2, $Ax3);
    fn const_permute(self) -> Self::Permuted {
        let mut result: <Self::Permuted as Tensor>::NoTape = TensorCreator::zeros();
        <Cpu as DevicePermute4<_, _, $Ax0, $Ax1, $Ax2, $Ax3>>::permute(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            <Cpu as DevicePermute4<_, _, $Ax0, $Ax1, $Ax2, $Ax3>>::inverse_permute(t.mut_data(), result_grad);
            Cpu::add(t_grad, t.data());
        })
    }
}
    };
}

macro_rules! permutations {
    ([$Ax0:tt, $Ax1:tt]) => {
        impl_permute!($Ax0, $Ax1);
        impl_permute!($Ax1, $Ax0);
    };

    ([$Ax0:tt, $Ax1:tt, $Ax2:tt]) => {
        permutations!($Ax0, [$Ax1, $Ax2]);
        permutations!($Ax1, [$Ax0, $Ax2]);
        permutations!($Ax2, [$Ax0, $Ax1]);
    };
    ($Ax0:tt, [$Ax1:tt, $Ax2:tt]) => {
        impl_permute!($Ax0, $Ax1, $Ax2);
        impl_permute!($Ax0, $Ax2, $Ax1);
    };

    ([$Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt]) => {
        permutations!($Ax0, [$Ax1, $Ax2, $Ax3]);
        permutations!($Ax1, [$Ax0, $Ax2, $Ax3]);
        permutations!($Ax2, [$Ax0, $Ax1, $Ax3]);
        permutations!($Ax3, [$Ax0, $Ax1, $Ax2]);
    };
    ($Ax0:tt, [$Ax1:tt, $Ax2:tt, $Ax3:tt]) => {
        permutations!($Ax0, $Ax1, [$Ax2, $Ax3]);
        permutations!($Ax0, $Ax2, [$Ax1, $Ax3]);
        permutations!($Ax0, $Ax3, [$Ax1, $Ax2]);
    };
    ($Ax0:tt, $Ax1:tt, [$Ax2:tt, $Ax3:tt]) => {
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax3);
        impl_permute!($Ax0, $Ax1, $Ax3, $Ax2);
    };
}

permutations!([0, 1]);
permutations!([0, 1, 2]);
permutations!([0, 1, 2, 3]);

pub trait Permute2Sugar: Sized {
    fn permute_axes<const I: isize, const J: isize>(self) -> Self::Permuted
    where
        Self: ConstPermute2<I, J>,
    {
        ConstPermute2::<I, J>::const_permute(self)
    }
}
impl<const M: usize, const N: usize, H> Permute2Sugar for Tensor2D<M, N, H> {}

pub trait Permute3Sugar: Sized {
    fn permute_axes<const I: isize, const J: isize, const K: isize>(self) -> Self::Permuted
    where
        Self: ConstPermute3<I, J, K>,
    {
        ConstPermute3::<I, J, K>::const_permute(self)
    }
}
impl<const M: usize, const N: usize, const O: usize, H> Permute3Sugar for Tensor3D<M, N, O, H> {}

pub trait Permute4Sugar: Sized {
    fn permute_axes<const I: isize, const J: isize, const K: isize, const L: isize>(
        self,
    ) -> Self::Permuted
    where
        Self: ConstPermute4<I, J, K, L>,
    {
        ConstPermute4::<I, J, K, L>::const_permute(self)
    }
}
impl<const M: usize, const N: usize, const O: usize, const P: usize, H> Permute4Sugar
    for Tensor4D<M, N, O, P, H>
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_permute_2d() {
        let mut rng = thread_rng();
        let t: Tensor2D<2, 3> = TensorCreator::randn(&mut rng);
        let r = t.clone().permute_axes::<1, 0>();
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(r.data()[j][i], t.data()[i][j]);
            }
        }
    }

    #[test]
    fn test_permute_3d() {
        let mut rng = thread_rng();
        let t: Tensor3D<3, 5, 7> = TensorCreator::randn(&mut rng);
        let r = t.clone().permute_axes::<1, 2, 0>();
        for i in 0..3 {
            for j in 0..5 {
                for k in 0..7 {
                    assert_eq!(r.data()[j][k][i], t.data()[i][j][k]);
                }
            }
        }
    }

    #[test]
    fn test_permute_4d() {
        let mut rng = thread_rng();
        let t: Tensor4D<3, 5, 7, 9> = TensorCreator::randn(&mut rng);
        let r = t.clone().permute_axes::<1, 3, 0, 2>();
        for i in 0..3 {
            for j in 0..5 {
                for k in 0..7 {
                    for l in 0..9 {
                        assert_eq!(r.data()[j][l][i][k], t.data()[i][j][k][l]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_permute_2d_backwards() {
        let mut rng = thread_rng();
        let t: Tensor2D<3, 6> = TensorCreator::randn(&mut rng);
        let g1 = t.trace().permute_axes::<1, 0>().exp().sum().backward();
        let g2 = t.trace().exp().sum().backward();
        assert_eq!(g1.ref_gradient(&t), g2.ref_gradient(&t));
    }

    #[test]
    fn test_permute_3d_backwards() {
        let mut rng = thread_rng();
        let t: Tensor3D<3, 6, 9> = TensorCreator::randn(&mut rng);
        let g1 = t.trace().permute_axes::<1, 2, 0>().exp().sum().backward();
        let g2 = t.trace().exp().sum().backward();
        assert_eq!(g1.ref_gradient(&t), g2.ref_gradient(&t));
    }

    #[test]
    fn test_permute_4d_backwards() {
        let mut rng = thread_rng();
        let t: Tensor4D<3, 6, 9, 11> = TensorCreator::randn(&mut rng);
        let g1 = t
            .trace()
            .permute_axes::<3, 1, 0, 2>()
            .exp()
            .sum()
            .backward();
        let g2 = t.trace().exp().sum().backward();
        assert_eq!(g1.ref_gradient(&t), g2.ref_gradient(&t));
    }

    #[test]
    fn test_valid_permutations() {
        let _ = <Tensor2D<3, 5> as ConstPermute2<0, 1>>::const_permute;
        let _ = <Tensor2D<3, 5> as ConstPermute2<1, 0>>::const_permute;

        let _ = <Tensor3D<3, 5, 7> as ConstPermute3<0, 1, 2>>::const_permute;
        let _ = <Tensor3D<3, 5, 7> as ConstPermute3<0, 2, 1>>::const_permute;
        let _ = <Tensor3D<3, 5, 7> as ConstPermute3<1, 0, 2>>::const_permute;
        let _ = <Tensor3D<3, 5, 7> as ConstPermute3<1, 2, 0>>::const_permute;
        let _ = <Tensor3D<3, 5, 7> as ConstPermute3<2, 0, 1>>::const_permute;
        let _ = <Tensor3D<3, 5, 7> as ConstPermute3<2, 1, 0>>::const_permute;

        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<0, 1, 2, 3>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<0, 1, 3, 2>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<0, 2, 1, 3>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<0, 2, 3, 1>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<0, 3, 2, 1>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<0, 3, 1, 2>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<1, 0, 2, 3>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<1, 0, 3, 2>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<1, 2, 0, 3>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<1, 2, 3, 0>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<1, 3, 0, 2>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<1, 3, 2, 0>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<2, 0, 1, 3>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<2, 0, 3, 1>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<2, 1, 0, 3>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<2, 1, 3, 0>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<2, 3, 0, 1>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<2, 3, 1, 0>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<3, 0, 1, 2>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<3, 0, 2, 1>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<3, 1, 0, 2>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<3, 1, 2, 0>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<3, 2, 0, 1>>::const_permute;
        let _ = <Tensor4D<3, 5, 7, 9> as ConstPermute4<3, 2, 1, 0>>::const_permute;
    }
}
