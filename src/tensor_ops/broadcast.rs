//! TODO describe implementation details

use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

/// TODO
pub trait Broadcast1<T, const I: isize> {
    /// TODO
    fn broadcast1(self) -> T;
}

/// TODO
pub trait Broadcast2<T, const I1: isize, const I2: isize> {
    /// TODO
    fn broadcast2(self) -> T;
}

/// TODO
pub trait Broadcast3<T, const I1: isize, const I2: isize, const I3: isize> {
    /// TODO
    fn broadcast3(self) -> T;
}

/// TODO
pub trait Broadcast4<T, const I1: isize, const I2: isize, const I3: isize, const I4: isize> {
    /// TODO
    fn broadcast4(self) -> T;
}

macro_rules! impl_broadcast {
    (
        $SrcTy:tt, [$($SrcDims:tt),*], $DstTy:tt, [$($DstDims:tt),*],
        $TensorTrait:tt, $fn_name:tt, [$($Axes:expr),*], [$($BDims:tt),*], $DeviceTrait:tt, {$($Dims:tt),*}
    ) => {
impl<$(const $Dims: usize, )* H: Tape> $TensorTrait<$DstTy<$($DstDims, )* H>, $($Axes, )*> for $SrcTy<$($SrcDims, )* H> {
    fn $fn_name(self) -> $DstTy<$($DstDims, )* H> {
        let mut result = $DstTy::zeros();
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

// #[rustfmt::skip]
// impl_broadcast!(Tensor0D, [], Tensor1D, [M], ConstBroadcast1, broadcast1, [0], [M], ForEachBroadcast1, {M});
#[rustfmt::skip]
impl_broadcast!(Tensor0D, [], Tensor1D, [M], Broadcast1, broadcast1, [-1], [M], ForEachBroadcast1, {M});

impl_broadcast!(Tensor0D, [], Tensor2D, [M, N], Broadcast2, broadcast2, [0, 1], [M, N], ForEachBroadcast2, {M, N});
impl_broadcast!(Tensor0D, [], Tensor3D, [M, N, O], Broadcast3, broadcast3, [0, 1, 2], [M, N, O], ForEachBroadcast3, {M, N, O});
impl_broadcast!(Tensor0D, [], Tensor4D, [M, N, O, P], Broadcast4, broadcast4, [0, 1, 2, 3], [M, N, O, P], ForEachBroadcast4, {M, N, O, P});

// impl_broadcast!(Tensor1D, [M], Tensor2D, [M, N], ConstBroadcast1, broadcast1, [1], [N], ForEachBroadcast1, {M, N});
impl_broadcast!(Tensor1D, [M], Tensor2D, [M, N], Broadcast1, broadcast1, [-1], [N], ForEachBroadcast1, {M, N});
impl_broadcast!(Tensor1D, [N], Tensor2D, [M, N], Broadcast1, broadcast1, [0], [M], ForEachBroadcast1, {M, N});
impl_broadcast!(Tensor1D, [M], Tensor3D, [M, N, O], Broadcast2, broadcast2, [1, 2], [N, O], ForEachBroadcast2, {M, N, O});
impl_broadcast!(Tensor1D, [N], Tensor3D, [M, N, O], Broadcast2, broadcast2, [0, 2], [M, O], ForEachBroadcast2, {M, N, O});
impl_broadcast!(Tensor1D, [O], Tensor3D, [M, N, O], Broadcast2, broadcast2, [0, 1], [M, N], ForEachBroadcast2, {M, N, O});
impl_broadcast!(Tensor1D, [M], Tensor4D, [M, N, O, P], Broadcast3, broadcast3, [1, 2, 3], [N, O, P], ForEachBroadcast3, {M, N, O, P});
impl_broadcast!(Tensor1D, [N], Tensor4D, [M, N, O, P], Broadcast3, broadcast3, [0, 2, 3], [M, O, P], ForEachBroadcast3, {M, N, O, P});
impl_broadcast!(Tensor1D, [O], Tensor4D, [M, N, O, P], Broadcast3, broadcast3, [0, 1, 3], [M, N, P], ForEachBroadcast3, {M, N, O, P});
impl_broadcast!(Tensor1D, [P], Tensor4D, [M, N, O, P], Broadcast3, broadcast3, [0, 1, 2], [M, N, O], ForEachBroadcast3, {M, N, O, P});

// impl_broadcast!(Tensor2D, [M, N], Tensor3D, [M, N, O], ConstBroadcast1, broadcast1, [2], [O], ForEachBroadcast1, {M, N, O});
impl_broadcast!(Tensor2D, [M, N], Tensor3D, [M, N, O], Broadcast1, broadcast1, [-1], [O], ForEachBroadcast1, {M, N, O});
impl_broadcast!(Tensor2D, [M, O], Tensor3D, [M, N, O], Broadcast1, broadcast1, [1], [N], ForEachBroadcast1, {M, N, O});
impl_broadcast!(Tensor2D, [N, O], Tensor3D, [M, N, O], Broadcast1, broadcast1, [0], [M], ForEachBroadcast1, {M, N, O});
impl_broadcast!(Tensor2D, [M, N], Tensor4D, [M, N, O, P], Broadcast2, broadcast2, [2, 3], [O, P], ForEachBroadcast2, {M, N, O, P});
impl_broadcast!(Tensor2D, [M, O], Tensor4D, [M, N, O, P], Broadcast2, broadcast2, [1, 3], [N, P], ForEachBroadcast2, {M, N, O, P});
impl_broadcast!(Tensor2D, [M, P], Tensor4D, [M, N, O, P], Broadcast2, broadcast2, [1, 2], [N, O], ForEachBroadcast2, {M, N, O, P});
impl_broadcast!(Tensor2D, [N, O], Tensor4D, [M, N, O, P], Broadcast2, broadcast2, [0, 3], [M, P], ForEachBroadcast2, {M, N, O, P});
impl_broadcast!(Tensor2D, [N, P], Tensor4D, [M, N, O, P], Broadcast2, broadcast2, [0, 2], [M, O], ForEachBroadcast2, {M, N, O, P});
impl_broadcast!(Tensor2D, [O, P], Tensor4D, [M, N, O, P], Broadcast2, broadcast2, [0, 1], [M, N], ForEachBroadcast2, {M, N, O, P});

// impl_broadcast!(Tensor3D, [M, N, O], Tensor4D, [M, N, O, P], ConstBroadcast1, broadcast1, [3], [P], ForEachBroadcast1, {M, N, O, P});
impl_broadcast!(Tensor3D, [M, N, O], Tensor4D, [M, N, O, P], Broadcast1, broadcast1, [-1], [P], ForEachBroadcast1, {M, N, O, P});
impl_broadcast!(Tensor3D, [M, N, P], Tensor4D, [M, N, O, P], Broadcast1, broadcast1, [2], [O], ForEachBroadcast1, {M, N, O, P});
impl_broadcast!(Tensor3D, [M, O, P], Tensor4D, [M, N, O, P], Broadcast1, broadcast1, [1], [N], ForEachBroadcast1, {M, N, O, P});
impl_broadcast!(Tensor3D, [N, O, P], Tensor4D, [M, N, O, P], Broadcast1, broadcast1, [0], [M], ForEachBroadcast1, {M, N, O, P});

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
