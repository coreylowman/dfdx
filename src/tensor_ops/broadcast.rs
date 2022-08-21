use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

/// Broadcasts the `I`th dimension. Increases number dimensions by 1. Results in `T`. Opposite of [Reduce1].
pub trait Broadcast1<T, const I: isize> {
    /// Broadcast `self` into `T`, increasing number dimensions by 1.
    ///
    /// Examples:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let _: Tensor1D<5> = Tensor0D::zeros().broadcast1();
    ///
    /// // broadcast the 0th axis
    /// let _: Tensor2D<5, 3> = Tensor1D::<3>::zeros().broadcast1();
    ///
    /// // broadcast the 1st axis into a 3d tensor
    /// let _: Tensor3D<3, 5, 7> = Tensor2D::<3, 7>::zeros().broadcast1();
    ///
    /// // broadcast the last axis into 4d tensor
    /// let _: Tensor4D<3, 5, 7, 9> = Tensor3D::<3, 5, 7>::zeros().broadcast1();
    /// ```
    fn broadcast1(self) -> T;
}

/// Broadcasts dimensions `I1` and `I2`. Increases number dimensions by 2. Results in `T`.
pub trait Broadcast2<T, const I1: isize, const I2: isize> {
    /// Broadcast `self` into `T`, increasing number dimensions by 2.
    ///
    /// Examples:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let _: Tensor2D<3, 5> = Tensor0D::zeros().broadcast2();
    ///
    /// // broadcast the 1st & 2nd axis
    /// let _: Tensor2D<3, 5, 7> = Tensor1D::<3>::zeros().broadcast2();
    ///
    /// // broadcast the 0th & 2nd axis
    /// let _: Tensor2D<5, 3, 7> = Tensor1D::<3>::zeros().broadcast2();
    ///
    /// // broadcast the 0th & 1st axis
    /// let _: Tensor2D<7, 5, 3> = Tensor1D::<3>::zeros().broadcast2();
    /// ```
    fn broadcast2(self) -> T;
}

/// Broadcasts dimensions `I1`, `I2`, and `I3`. Increases number dimensions by 3. Results in `T`.
pub trait Broadcast3<T, const I1: isize, const I2: isize, const I3: isize> {
    /// Broadcast `self` into `T`, increasing number dimensions by 3.
    ///
    /// Examples:
    /// ```rust
    /// # use dfdx::prelude::*;
    ///
    /// // broadcast axes 1, 2, 3
    /// let _: Tensor3D<3, 5, 7, 9> = Tensor1D::<3>::zeros().broadcast3();
    ///
    /// // broadcast axes 0, 2, 3
    /// let _: Tensor3D<9, 3, 5, 7> = Tensor1D::<3>::zeros().broadcast3();
    ///
    /// // broadcast axes 0, 1, 3
    /// let _: Tensor3D<9, 7, 3, 5> = Tensor1D::<3>::zeros().broadcast3();
    ///
    /// // braodcast axes 0, 1, 2
    /// let _: Tensor3D<9, 7, 5, 3> = Tensor1D::<3>::zeros().broadcast3();
    /// ```
    fn broadcast3(self) -> T;
}

/// Broadcasts dimensions `I1`, `I2`, `I3`, and `I4`. Increases number dimensions by 4. Results in `T`.
pub trait Broadcast4<T, const I1: isize, const I2: isize, const I3: isize, const I4: isize> {
    /// Broadcast `self` into `T`, increasing number dimensions by 4.
    ///
    /// Examples:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let _: Tensor4D<1, 2, 3, 4> = Tensor0D::zeros().broadcast4();
    /// ```
    fn broadcast4(self) -> T;
}

macro_rules! impl_broadcast {
    (
        [$($Axes:expr),*],
        $SrcTy:ty, $DstTy:ty,
        $TensorTrait:tt, $fn_name:tt,
        $DeviceTrait:tt, {$($Dims:tt),*}
    ) => {
impl<$(const $Dims: usize, )* H: Tape> $TensorTrait<$DstTy, $($Axes, )*> for $SrcTy {
    fn $fn_name(self) -> $DstTy {
        let mut result = <$DstTy as Tensor>::NoTape::zeros();
        <Cpu as $DeviceTrait<_, _, $($Axes),*>>::broadcast_copy(result.mut_data(), self.data());
        move_tape_and_add_backward_op(self, result, move |t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            <Cpu as $DeviceTrait<_, _, $($Axes),*>>::broadcast_add(t_grad, result_grad);
        })
    }
}
    };
}

impl<H: Tape> Broadcast1<Tensor0D<H>, -1> for Tensor0D<H> {
    fn broadcast1(self) -> Tensor0D<H> {
        self
    }
}

// 0d -> Nd
#[rustfmt::skip]
impl_broadcast!([-1], Tensor0D<H>, Tensor1D<M, H>, Broadcast1, broadcast1, ForEachBroadcast1, {M});
impl_broadcast!([0, 1], Tensor0D<H>, Tensor2D<M, N, H>, Broadcast2, broadcast2, ForEachBroadcast2, {M, N});
impl_broadcast!([0, 1, 2], Tensor0D<H>, Tensor3D<M, N, O, H>, Broadcast3, broadcast3, ForEachBroadcast3, {M, N, O});
impl_broadcast!([0, 1, 2, 3], Tensor0D<H>, Tensor4D<M, N, O, P, H>, Broadcast4, broadcast4, ForEachBroadcast4, {M, N, O, P});

// 1d -> Nd
impl_broadcast!([-1], Tensor1D<M, H>, Tensor2D<M, N, H>, Broadcast1, broadcast1, ForEachBroadcast1, {M, N});
impl_broadcast!([0], Tensor1D<N, H>, Tensor2D<M, N, H>, Broadcast1, broadcast1, ForEachBroadcast1, {M, N});
impl_broadcast!([1, 2], Tensor1D<M, H>, Tensor3D<M, N, O, H>, Broadcast2, broadcast2, ForEachBroadcast2, {M, N, O});
impl_broadcast!([0, 2], Tensor1D<N, H>, Tensor3D<M, N, O, H>, Broadcast2, broadcast2, ForEachBroadcast2, {M, N, O});
impl_broadcast!([0, 1], Tensor1D<O, H>, Tensor3D<M, N, O, H>, Broadcast2, broadcast2, ForEachBroadcast2, {M, N, O});
impl_broadcast!([1, 2, 3], Tensor1D<M, H>, Tensor4D<M, N, O, P, H>, Broadcast3, broadcast3, ForEachBroadcast3, {M, N, O, P});
impl_broadcast!([0, 2, 3], Tensor1D<N, H>, Tensor4D<M, N, O, P, H>, Broadcast3, broadcast3, ForEachBroadcast3, {M, N, O, P});
impl_broadcast!([0, 1, 3], Tensor1D<O, H>, Tensor4D<M, N, O, P, H>, Broadcast3, broadcast3, ForEachBroadcast3, {M, N, O, P});
impl_broadcast!([0, 1, 2], Tensor1D<P, H>, Tensor4D<M, N, O, P, H>, Broadcast3, broadcast3, ForEachBroadcast3, {M, N, O, P});

// 2d -> Nd
impl_broadcast!([-1], Tensor2D<M, N, H>, Tensor3D<M, N, O, H>, Broadcast1, broadcast1, ForEachBroadcast1, {M, N, O});
impl_broadcast!([1], Tensor2D<M, O, H>, Tensor3D<M, N, O, H>, Broadcast1, broadcast1, ForEachBroadcast1, {M, N, O});
impl_broadcast!([0], Tensor2D<N, O, H>, Tensor3D<M, N, O, H>, Broadcast1, broadcast1, ForEachBroadcast1, {M, N, O});
impl_broadcast!([2, 3], Tensor2D<M, N, H>, Tensor4D<M, N, O, P, H>, Broadcast2, broadcast2, ForEachBroadcast2, {M, N, O, P});
impl_broadcast!([1, 3], Tensor2D<M, O, H>, Tensor4D<M, N, O, P, H>, Broadcast2, broadcast2, ForEachBroadcast2, {M, N, O, P});
impl_broadcast!([1, 2], Tensor2D<M, P, H>, Tensor4D<M, N, O, P, H>, Broadcast2, broadcast2, ForEachBroadcast2, {M, N, O, P});
impl_broadcast!([0, 3], Tensor2D<N, O, H>, Tensor4D<M, N, O, P, H>, Broadcast2, broadcast2, ForEachBroadcast2, {M, N, O, P});
impl_broadcast!([0, 2], Tensor2D<N, P, H>, Tensor4D<M, N, O, P, H>, Broadcast2, broadcast2, ForEachBroadcast2, {M, N, O, P});
impl_broadcast!([0, 1], Tensor2D<O, P, H>, Tensor4D<M, N, O, P, H>, Broadcast2, broadcast2, ForEachBroadcast2, {M, N, O, P});

// 3d -> 4d
impl_broadcast!([-1], Tensor3D<M, N, O, H>, Tensor4D<M, N, O, P, H>, Broadcast1, broadcast1, ForEachBroadcast1, {M, N, O, P});
impl_broadcast!([2], Tensor3D<M, N, P, H>, Tensor4D<M, N, O, P, H>, Broadcast1, broadcast1, ForEachBroadcast1, {M, N, O, P});
impl_broadcast!([1], Tensor3D<M, O, P, H>, Tensor4D<M, N, O, P, H>, Broadcast1, broadcast1, ForEachBroadcast1, {M, N, O, P});
impl_broadcast!([0], Tensor3D<N, O, P, H>, Tensor4D<M, N, O, P, H>, Broadcast1, broadcast1, ForEachBroadcast1, {M, N, O, P});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::thread_rng;

    #[test]
    fn test_valid_1d_broadcasts() {
        let _: Tensor1D<5> = Tensor0D::zeros().broadcast1();

        let _: Tensor2D<5, 3> = Tensor1D::<3>::zeros().broadcast1();
        let _: Tensor2D<5, 3> = Tensor1D::<5>::zeros().broadcast1();
        let _: Tensor2D<5, 3> = Tensor1D::<5>::zeros().broadcast1();

        let _: Tensor3D<3, 5, 7> = Tensor2D::<5, 7>::zeros().broadcast1();
        let _: Tensor3D<3, 5, 7> = Tensor2D::<3, 7>::zeros().broadcast1();
        let _: Tensor3D<3, 5, 7> = Tensor2D::<3, 5>::zeros().broadcast1();
        let _: Tensor3D<3, 5, 7> = Tensor2D::<3, 5>::zeros().broadcast1();

        let _: Tensor4D<3, 5, 7, 9> = Tensor3D::<5, 7, 9>::zeros().broadcast1();
        let _: Tensor4D<3, 5, 7, 9> = Tensor3D::<3, 7, 9>::zeros().broadcast1();
        let _: Tensor4D<3, 5, 7, 9> = Tensor3D::<3, 5, 9>::zeros().broadcast1();
        let _: Tensor4D<3, 5, 7, 9> = Tensor3D::<3, 5, 7>::zeros().broadcast1();
        let _: Tensor4D<3, 5, 7, 9> = Tensor3D::<3, 5, 7>::zeros().broadcast1();
    }

    #[test]
    fn test_valid_2d_broadcasts() {
        let _: Tensor2D<5, 3> = Tensor0D::zeros().broadcast2();

        let _: Tensor3D<3, 5, 7> = Tensor1D::<3>::zeros().broadcast2();
        let _: Tensor3D<3, 5, 7> = Tensor1D::<5>::zeros().broadcast2();
        let _: Tensor3D<3, 5, 7> = Tensor1D::<7>::zeros().broadcast2();

        let _: Tensor4D<3, 5, 7, 9> = Tensor2D::<3, 5>::zeros().broadcast2();
        let _: Tensor4D<3, 5, 7, 9> = Tensor2D::<3, 7>::zeros().broadcast2();
        let _: Tensor4D<3, 5, 7, 9> = Tensor2D::<3, 9>::zeros().broadcast2();
        let _: Tensor4D<3, 5, 7, 9> = Tensor2D::<5, 7>::zeros().broadcast2();
        let _: Tensor4D<3, 5, 7, 9> = Tensor2D::<5, 9>::zeros().broadcast2();
        let _: Tensor4D<3, 5, 7, 9> = Tensor2D::<7, 9>::zeros().broadcast2();
    }

    #[test]
    fn test_valid_3d_broadcasts() {
        let _: Tensor3D<3, 5, 7> = Tensor0D::zeros().broadcast3();

        let _: Tensor4D<3, 5, 7, 9> = Tensor1D::<3>::zeros().broadcast3();
        let _: Tensor4D<3, 5, 7, 9> = Tensor1D::<5>::zeros().broadcast3();
        let _: Tensor4D<3, 5, 7, 9> = Tensor1D::<7>::zeros().broadcast3();
        let _: Tensor4D<3, 5, 7, 9> = Tensor1D::<9>::zeros().broadcast3();
    }

    #[test]
    fn test_broadcast_backwards() {
        let mut rng = thread_rng();
        let a: Tensor1D<3> = TensorCreator::randn(&mut rng);
        let b: Tensor2D<5, 3> = TensorCreator::randn(&mut rng);
        let a_up: Tensor2D<5, 3, OwnedTape> = a.trace().broadcast1();
        assert_eq!(a_up.data(), &[*a.data(); 5]);
        let r = mul(a_up, &b);
        let g = r.exp().mean().backward();
        // a's gradient: (b * (b * a).exp()).sum(0) / 15
        // b's gradient: (a * (b * a).exp()) / 15
        let a_up: Tensor2D<5, 3> = a.clone().broadcast1();
        let a_grad = mul(mul(b.clone(), &a_up).exp(), &b).sum_axis::<0>() / 15.0;
        let b_grad = mul(mul(b.clone(), &a_up).exp(), &a_up) / 15.0;
        assert_close(g.ref_gradient(&a), a_grad.data());
        assert_close(g.ref_gradient(&b), b_grad.data());
    }
}
