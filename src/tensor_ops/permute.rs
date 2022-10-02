//! Implementation of permute operation
//!
//! There are two traits used here:
//! 1. [Permute]
//! 2. [Permute2D], [Permute3D], and [Permute4D]
//!
//! The [Permute] versions have the axes specified on the
//! trait itself (i.e. `Permute::<Axes2<1, 0>>::permute()`).
//!
//! [Permute2D], [Permute3D], and [Permute4D] have the axes specified on the
//! function inside the trait (i.e. `Permute2D::permute_axis::<1, 0>()`)

use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

/// Permute axes that are specified at the trait level. See
/// [Permute2D], [Permute3D], and [Permute4D]
/// for a more ergonomic version.
pub trait Permute<Axes> {
    /// The resulting type after being permuted
    type Permuted;
    fn permute(self) -> Self::Permuted;
}

/// Permute 2d tensor with new axes order specified at the function call level.
pub trait Permute2D: Sized {
    /// Generics:
    /// - `I` the index of the new 1st dimension, can be 0 or 1
    /// - `J` the index of the new 2nd dimension, can be 0 or 1
    ///
    /// Examples
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let t: Tensor2D<2, 3> = TensorCreator::zeros();
    /// let r: Tensor2D<3, 2> = t.permute_axes::<1, 0>();
    /// ```
    fn permute_axes<const I: isize, const J: isize>(self) -> Self::Permuted
    where
        Self: Permute<Axes2<I, J>>,
    {
        self.permute()
    }
}
impl<const M: usize, const N: usize, H> Permute2D for Tensor2D<M, N, H> {}

/// Permute 3d tensor with new axes order specified at the function call level.
pub trait Permute3D: Sized {
    /// Generics:
    /// - `I` the index of the new 1st dimension, can be 0, 1, or 2
    /// - `J` the index of the new 2nd dimension, can be 0, 1, or 2
    /// - `K` the index of the new 3rd dimension, can be 0, 1, or 2
    ///
    /// Examples
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let t: Tensor3D<3, 5, 7> = TensorCreator::zeros();
    /// let _: Tensor3D<5, 7, 3> = t.clone().permute_axes::<1, 2, 0>();
    /// let _: Tensor3D<7, 3, 5> = t.clone().permute_axes::<2, 0, 1>();
    /// ```
    fn permute_axes<const I: isize, const J: isize, const K: isize>(self) -> Self::Permuted
    where
        Self: Permute<Axes3<I, J, K>>,
    {
        self.permute()
    }
}
impl<const M: usize, const N: usize, const O: usize, H> Permute3D for Tensor3D<M, N, O, H> {}

/// Permute 4d tensor with new axes order specified at the function call level.
pub trait Permute4D: Sized {
    /// Generics:
    /// - `I` the index of the new 1st dimension, can be 0, 1, 2, or 3
    /// - `J` the index of the new 2nd dimension, can be 0, 1, 2, or 3
    /// - `K` the index of the new 3rd dimension, can be 0, 1, 2, or 3
    /// - `L` the index of the new 4th dimension, can be 0, 1, 2, or 3
    ///
    /// Examples
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let t: Tensor4D<3, 5, 7, 9> = TensorCreator::zeros();
    /// let _: Tensor4D<9, 5, 7, 3> = t.clone().permute_axes::<3, 1, 2, 0>();
    /// let _: Tensor4D<7, 3, 9, 5> = t.clone().permute_axes::<2, 0, 3, 1>();
    /// let _: Tensor4D<5, 9, 3, 7> = t.clone().permute_axes::<1, 3, 0, 2>();
    /// ```
    fn permute_axes<const I: isize, const J: isize, const K: isize, const L: isize>(
        self,
    ) -> Self::Permuted
    where
        Self: Permute<Axes4<I, J, K, L>>,
    {
        self.permute()
    }
}
impl<const M: usize, const N: usize, const O: usize, const P: usize, H> Permute4D
    for Tensor4D<M, N, O, P, H>
{
}

/// Returns const generic for a specific axis.
#[rustfmt::skip]
macro_rules! axis { (0) => { M }; (1) => { N }; (2) => { O }; (3) => { P }; }

/// Helper macro that creates a tensor based on axes passed in.
/// E.g. `tensor!(2, 0, 1)` returns `Tensor3D<O, M, N, H>`
#[rustfmt::skip]
macro_rules! tensor {
    ($Ax0:tt) => { Tensor1D<axis!($Ax0), H> };
    ($Ax0:tt, $Ax1:tt) => { Tensor2D<axis!($Ax0), axis!($Ax1), H> };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt) => { Tensor3D<axis!($Ax0), axis!($Ax1), axis!($Ax2), H> };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt) => { Tensor4D<axis!($Ax0), axis!($Ax1), axis!($Ax2), axis!($Ax3), H> };
}

/// Concrete implementations of permute for 2-4d tensors. These just call device level permute & inverse permute
/// functions.
#[rustfmt::skip]
macro_rules! impl_permute {
    ($Ax0:tt, $Ax1:tt) => {
impl<const M: usize, const N: usize, H: Tape> Permute<Axes2<$Ax0, $Ax1>> for tensor!(0, 1) {
    type Permuted = tensor!($Ax0, $Ax1);
    fn permute(self) -> Self::Permuted {
        let mut result: <Self::Permuted as Tensor>::NoTape = TensorCreator::zeros();
        <Cpu as DevicePermute<_, _, Axes2<$Ax0, $Ax1>>>::permute(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            <Cpu as DevicePermute<_, _, Axes2<$Ax0, $Ax1>>>::inverse_permute(t.mut_data(), result_grad);
            Cpu::add(t_grad, t.data());
        })
    }
}
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt) => {
impl<const M: usize, const N: usize, const O: usize, H: Tape> Permute<Axes3<$Ax0, $Ax1, $Ax2>> for tensor!(0, 1, 2) {
    type Permuted = tensor!($Ax0, $Ax1, $Ax2);
    fn permute(self) -> Self::Permuted {
        let mut result: <Self::Permuted as Tensor>::NoTape = TensorCreator::zeros();
        <Cpu as DevicePermute<_, _, Axes3<$Ax0, $Ax1, $Ax2>>>::permute(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            <Cpu as DevicePermute<_, _, Axes3<$Ax0, $Ax1, $Ax2>>>::inverse_permute(t.mut_data(), result_grad);
            Cpu::add(t_grad, t.data());
        })
    }
}
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt) => {
impl<const M: usize, const N: usize, const O: usize, const P: usize, H: Tape>
    Permute<Axes4<$Ax0, $Ax1, $Ax2, $Ax3>> for tensor!(0, 1, 2, 3)
{
    type Permuted = tensor!($Ax0, $Ax1, $Ax2, $Ax3);
    fn permute(self) -> Self::Permuted {
        let mut result: <Self::Permuted as Tensor>::NoTape = TensorCreator::zeros();
        <Cpu as DevicePermute<_, _, Axes4<$Ax0, $Ax1, $Ax2, $Ax3>>>::permute(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            <Cpu as DevicePermute<_, _, Axes4<$Ax0, $Ax1, $Ax2, $Ax3>>>::inverse_permute(t.mut_data(), result_grad);
            Cpu::add(t_grad, t.data());
        })
    }
}
    };
}

/// Expands all the possible permutations of 2-4 elements.
/// Expands [impl_permute!] at the base level.
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
        let g1 = backward(t.trace().permute_axes::<1, 0>().exp().sum());
        let g2 = backward(t.trace().exp().sum());
        assert_eq!(g1.ref_gradient(&t), g2.ref_gradient(&t));
    }

    #[test]
    fn test_permute_3d_backwards() {
        let mut rng = thread_rng();
        let t: Tensor3D<3, 6, 9> = TensorCreator::randn(&mut rng);
        let g1 = backward(t.trace().permute_axes::<1, 2, 0>().exp().sum());
        let g2 = backward(t.trace().exp().sum());
        assert_eq!(g1.ref_gradient(&t), g2.ref_gradient(&t));
    }

    #[test]
    fn test_permute_4d_backwards() {
        let mut rng = thread_rng();
        let t: Tensor4D<3, 6, 9, 11> = TensorCreator::randn(&mut rng);
        let g1 = backward(t.trace().permute_axes::<3, 1, 0, 2>().exp().sum());
        let g2 = backward(t.trace().exp().sum());
        assert_eq!(g1.ref_gradient(&t), g2.ref_gradient(&t));
    }

    #[test]
    fn test_valid_permutations() {
        let _ = <Tensor2D<3, 5> as Permute<Axes2<0, 1>>>::permute;
        let _ = <Tensor2D<3, 5> as Permute<Axes2<1, 0>>>::permute;

        let _ = <Tensor3D<3, 5, 7> as Permute<Axes3<0, 1, 2>>>::permute;
        let _ = <Tensor3D<3, 5, 7> as Permute<Axes3<0, 2, 1>>>::permute;
        let _ = <Tensor3D<3, 5, 7> as Permute<Axes3<1, 0, 2>>>::permute;
        let _ = <Tensor3D<3, 5, 7> as Permute<Axes3<1, 2, 0>>>::permute;
        let _ = <Tensor3D<3, 5, 7> as Permute<Axes3<2, 0, 1>>>::permute;
        let _ = <Tensor3D<3, 5, 7> as Permute<Axes3<2, 1, 0>>>::permute;

        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<0, 1, 2, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<0, 1, 3, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<0, 2, 1, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<0, 2, 3, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<0, 3, 2, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<0, 3, 1, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<1, 0, 2, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<1, 0, 3, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<1, 2, 0, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<1, 2, 3, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<1, 3, 0, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<1, 3, 2, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<2, 0, 1, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<2, 0, 3, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<2, 1, 0, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<2, 1, 3, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<2, 3, 0, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<2, 3, 1, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<3, 0, 1, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<3, 0, 2, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<3, 1, 0, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<3, 1, 2, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<3, 2, 0, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as Permute<Axes4<3, 2, 1, 0>>>::permute;
    }
}
