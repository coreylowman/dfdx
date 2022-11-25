use super::utils::move_tape_and_add_backward_op;
use crate::devices::{Cpu, Device, DevicePermute};
use crate::gradients::Tape;
use crate::prelude::*;

/// Permutes self into `T` with the new order of axes specified via `Axes`.
pub trait PermuteTo<T, Axes> {
    /// Permutes the tensor
    ///
    /// Examples
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let _: Tensor2D<3, 2> = Tensor2D::<2, 3>::zeros().permute();
    /// let _: Tensor3D<3, 4, 2> = Tensor3D::<2, 3, 4>::zeros().permute();
    /// let _: Tensor4D<3, 4, 5, 2> = Tensor4D::<2, 3, 4, 5>::zeros().permute();
    /// let _: Tensor5D<3, 4, 5, 2, 1> = Tensor5D::<1, 2, 3, 4, 5>::zeros().permute();
    /// let _: Tensor6D<3, 4, 5, 2, 6, 1> = Tensor6D::<1, 2, 3, 4, 5, 6>::zeros().permute();
    /// ```
    fn permute(self) -> T;
}

/// Returns const generic for a specific axis.
#[rustfmt::skip]
macro_rules! axis { (0) => { M }; (1) => { N }; (2) => { O }; (3) => { P }; (4) => { Q }; (5) => { R }; }

/// Helper macro that creates a tensor based on axes passed in.
/// E.g. `tensor!(2, 0, 1)` returns `Tensor3D<O, M, N, H>`
#[rustfmt::skip]
macro_rules! tensor {
    ($Ax0:tt) => { Tensor1D<axis!($Ax0), H> };
    ($Ax0:tt, $Ax1:tt) => { Tensor2D<axis!($Ax0), axis!($Ax1), H> };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt) => { Tensor3D<axis!($Ax0), axis!($Ax1), axis!($Ax2), H> };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt) => { Tensor4D<axis!($Ax0), axis!($Ax1), axis!($Ax2), axis!($Ax3), H> };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt) => { Tensor5D<axis!($Ax0), axis!($Ax1), axis!($Ax2), axis!($Ax3), axis!($Ax4), H> };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt) => { Tensor6D<axis!($Ax0), axis!($Ax1), axis!($Ax2), axis!($Ax3), axis!($Ax4), axis!($Ax5), H> };
}

/// Concrete implementations of permute for 2-6d tensors. These just call device level permute & inverse permute
/// functions.
#[rustfmt::skip]
macro_rules! impl_permute {
    ($Ax0:tt, $Ax1:tt) => {
impl<const M: usize, const N: usize, H: Tape>
PermuteTo<tensor!($Ax0, $Ax1), Axes2<$Ax0, $Ax1>> for tensor!(0, 1)
{
    fn permute(self) -> tensor!($Ax0, $Ax1) {
        let mut result: <tensor!($Ax0, $Ax1) as Tensor>::NoTape = TensorCreator::zeros();
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
impl<const M: usize, const N: usize, const O: usize, H: Tape>
PermuteTo<tensor!($Ax0, $Ax1, $Ax2), Axes3<$Ax0, $Ax1, $Ax2>> for tensor!(0, 1, 2)
{
    fn permute(self) -> tensor!($Ax0, $Ax1, $Ax2) {
        let mut result: <tensor!($Ax0, $Ax1, $Ax2) as Tensor>::NoTape = TensorCreator::zeros();
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
PermuteTo<tensor!($Ax0, $Ax1, $Ax2, $Ax3), Axes4<$Ax0, $Ax1, $Ax2, $Ax3>> for tensor!(0, 1, 2, 3)
{
    fn permute(self) -> tensor!($Ax0, $Ax1, $Ax2, $Ax3) {
        let mut result: <tensor!($Ax0, $Ax1, $Ax2, $Ax3) as Tensor>::NoTape = TensorCreator::zeros();
        <Cpu as DevicePermute<_, _, Axes4<$Ax0, $Ax1, $Ax2, $Ax3>>>::permute(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            <Cpu as DevicePermute<_, _, Axes4<$Ax0, $Ax1, $Ax2, $Ax3>>>::inverse_permute(t.mut_data(), result_grad);
            Cpu::add(t_grad, t.data());
        })
    }
}
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt) => {
impl<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize, H: Tape>
PermuteTo<tensor!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4), Axes5<$Ax0, $Ax1, $Ax2, $Ax3, $Ax4>> for tensor!(0, 1, 2, 3, 4)
{
    fn permute(self) -> tensor!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4) {
        let mut result: <tensor!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4) as Tensor>::NoTape = TensorCreator::zeros();
        <Cpu as DevicePermute<_, _, Axes5<$Ax0, $Ax1, $Ax2, $Ax3, $Ax4>>>::permute(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            <Cpu as DevicePermute<_, _, Axes5<$Ax0, $Ax1, $Ax2, $Ax3, $Ax4>>>::inverse_permute(t.mut_data(), result_grad);
            Cpu::add(t_grad, t.data());
        })
    }
}
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt) => {
impl<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize, const R: usize, H: Tape>
PermuteTo<tensor!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5), Axes6<$Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5>> for tensor!(0, 1, 2, 3, 4, 5)
{
    fn permute(self) -> tensor!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5) {
        let mut result: <tensor!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5) as Tensor>::NoTape = TensorCreator::zeros();
        <Cpu as DevicePermute<_, _, Axes6<$Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5>>>::permute(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            <Cpu as DevicePermute<_, _, Axes6<$Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5>>>::inverse_permute(t.mut_data(), result_grad);
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

    ([$Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt]) => {
        permutations!($Ax0, [$Ax1, $Ax2, $Ax3, $Ax4]);
        permutations!($Ax1, [$Ax0, $Ax2, $Ax3, $Ax4]);
        permutations!($Ax2, [$Ax0, $Ax1, $Ax3, $Ax4]);
        permutations!($Ax3, [$Ax0, $Ax1, $Ax2, $Ax4]);
        permutations!($Ax4, [$Ax0, $Ax1, $Ax2, $Ax3]);
    };
    ($Ax0:tt, [$Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt]) => {
        permutations!($Ax0, $Ax1, [$Ax2, $Ax3, $Ax4]);
        permutations!($Ax0, $Ax2, [$Ax1, $Ax3, $Ax4]);
        permutations!($Ax0, $Ax3, [$Ax1, $Ax2, $Ax4]);
        permutations!($Ax0, $Ax4, [$Ax1, $Ax2, $Ax3]);
    };
    ($Ax0:tt, $Ax1:tt, [$Ax2:tt, $Ax3:tt, $Ax4:tt]) => {
        permutations!($Ax0, $Ax1, $Ax2, [$Ax3, $Ax4]);
        permutations!($Ax0, $Ax1, $Ax3, [$Ax2, $Ax4]);
        permutations!($Ax0, $Ax1, $Ax4, [$Ax2, $Ax3]);
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, [$Ax3:tt, $Ax4:tt]) => {
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4);
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax4, $Ax3);
    };

    ([$Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt]) => {
        permutations!($Ax0, [$Ax1, $Ax2, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax1, [$Ax0, $Ax2, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax2, [$Ax0, $Ax1, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax3, [$Ax0, $Ax1, $Ax2, $Ax4, $Ax5]);
        permutations!($Ax4, [$Ax0, $Ax1, $Ax2, $Ax3, $Ax5]);
        permutations!($Ax5, [$Ax0, $Ax1, $Ax2, $Ax3, $Ax4]);
    };
    ($Ax0:tt, [$Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt]) => {
        permutations!($Ax0, $Ax1, [$Ax2, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax2, [$Ax1, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax3, [$Ax1, $Ax2, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax4, [$Ax1, $Ax2, $Ax3, $Ax5]);
        permutations!($Ax0, $Ax5, [$Ax1, $Ax2, $Ax3, $Ax4]);
    };
    ($Ax0:tt, $Ax1:tt, [$Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt]) => {
        permutations!($Ax0, $Ax1, $Ax2, [$Ax3, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax3, [$Ax2, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax4, [$Ax2, $Ax3, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax5, [$Ax2, $Ax3, $Ax4]);
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, [$Ax3:tt, $Ax4:tt, $Ax5:tt]) => {
        permutations!($Ax0, $Ax1, $Ax2, $Ax3, [$Ax4, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax2, $Ax4, [$Ax3, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax2, $Ax5, [$Ax3, $Ax4]);
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, [$Ax4:tt, $Ax5:tt]) => {
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5);
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax3, $Ax5, $Ax4);
    };
}

permutations!([0, 1]);
permutations!([0, 1, 2]);
permutations!([0, 1, 2, 3]);
permutations!([0, 1, 2, 3, 4]);
permutations!([0, 1, 2, 3, 4, 5]);

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_permute_2d() {
        let mut rng = thread_rng();
        let t: Tensor2D<2, 3> = TensorCreator::randn(&mut rng);
        let r: Tensor2D<3, 2> = t.clone().permute();
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
        let r: Tensor3D<5, 7, 3> = t.clone().permute();
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
        let r: Tensor4D<5, 9, 3, 7> = t.clone().permute();
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
    fn test_permute_5d() {
        let mut rng = thread_rng();
        let t: Tensor5D<3, 5, 7, 9, 2> = TensorCreator::randn(&mut rng);
        let r: Tensor5D<5, 9, 2, 3, 7> = t.clone().permute();
        for i in 0..3 {
            for j in 0..5 {
                for k in 0..7 {
                    for l in 0..9 {
                        for h in 0..2 {
                            assert_eq!(r.data()[j][l][h][i][k], t.data()[i][j][k][l][h]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_permute_6d() {
        let mut rng = thread_rng();
        let t: Tensor6D<3, 5, 7, 9, 2, 1> = TensorCreator::randn(&mut rng);
        let r: Tensor6D<5, 1, 9, 2, 3, 7> = t.clone().permute();
        for i in 0..3 {
            for j in 0..5 {
                for k in 0..7 {
                    for l in 0..9 {
                        for h in 0..2 {
                            for n in 0..1 {
                                assert_eq!(r.data()[j][n][l][h][i][k], t.data()[i][j][k][l][h][n]);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_permute_2d_backwards() {
        let mut rng = thread_rng();
        let t: Tensor2D<3, 6> = TensorCreator::randn(&mut rng);
        let g1 = backward(PermuteTo::<_, Axes2<1, 0>>::permute(t.trace()).exp().sum());
        let g2 = backward(t.trace().exp().sum());
        assert_eq!(g1.ref_gradient(&t), g2.ref_gradient(&t));
    }

    #[test]
    fn test_permute_3d_backwards() {
        let mut rng = thread_rng();
        let t: Tensor3D<3, 6, 9> = TensorCreator::randn(&mut rng);
        let r: Tensor3D<3, 9, 6, _> = t.trace().permute();
        let g1 = backward(r.exp().sum());
        let g2 = backward(t.trace().exp().sum());
        assert_eq!(g1.ref_gradient(&t), g2.ref_gradient(&t));
    }

    #[test]
    fn test_permute_4d_backwards() {
        let mut rng = thread_rng();
        let t: Tensor4D<3, 6, 9, 11> = TensorCreator::randn(&mut rng);
        let r: Tensor4D<6, 3, 11, 9, _> = t.trace().permute();
        let g1 = backward(r.exp().sum());
        let g2 = backward(t.trace().exp().sum());
        assert_eq!(g1.ref_gradient(&t), g2.ref_gradient(&t));
    }

    #[test]
    fn test_valid_permutations() {
        let _ = <Tensor2D<3, 5> as PermuteTo<_, Axes2<0, 1>>>::permute;
        let _ = <Tensor2D<3, 5> as PermuteTo<_, Axes2<1, 0>>>::permute;

        let _ = <Tensor3D<3, 5, 7> as PermuteTo<_, Axes3<0, 1, 2>>>::permute;
        let _ = <Tensor3D<3, 5, 7> as PermuteTo<_, Axes3<0, 2, 1>>>::permute;
        let _ = <Tensor3D<3, 5, 7> as PermuteTo<_, Axes3<1, 0, 2>>>::permute;
        let _ = <Tensor3D<3, 5, 7> as PermuteTo<_, Axes3<1, 2, 0>>>::permute;
        let _ = <Tensor3D<3, 5, 7> as PermuteTo<_, Axes3<2, 0, 1>>>::permute;
        let _ = <Tensor3D<3, 5, 7> as PermuteTo<_, Axes3<2, 1, 0>>>::permute;

        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<0, 1, 2, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<0, 1, 3, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<0, 2, 1, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<0, 2, 3, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<0, 3, 2, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<0, 3, 1, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<1, 0, 2, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<1, 0, 3, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<1, 2, 0, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<1, 2, 3, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<1, 3, 0, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<1, 3, 2, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<2, 0, 1, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<2, 0, 3, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<2, 1, 0, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<2, 1, 3, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<2, 3, 0, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<2, 3, 1, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<3, 0, 1, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<3, 0, 2, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<3, 1, 0, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<3, 1, 2, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<3, 2, 0, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9> as PermuteTo<_, Axes4<3, 2, 1, 0>>>::permute;
    }
}
