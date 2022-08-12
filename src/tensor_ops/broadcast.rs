use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

pub trait ConstBroadcast1<const I: isize, const M: usize> {
    type Broadcasted;
    fn const_broadcast(self) -> Self::Broadcasted;
}

pub trait ConstBroadcast2<const I1: isize, const I2: isize, const M: usize, const N: usize> {
    type Broadcasted;
    fn const_broadcast(self) -> Self::Broadcasted;
}

pub trait ConstBroadcast3<
    const I1: isize,
    const I2: isize,
    const I3: isize,
    const M: usize,
    const N: usize,
    const O: usize,
>
{
    type Broadcasted;
    fn const_broadcast(self) -> Self::Broadcasted;
}

pub trait ConstBroadcast4<
    const I1: isize,
    const I2: isize,
    const I3: isize,
    const I4: isize,
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
>
{
    type Broadcasted;
    fn const_broadcast(self) -> Self::Broadcasted;
}

macro_rules! impl_broadcast {
    (
        $SrcTy:tt, [$($SrcDims:tt),*], $DstTy:tt, [$($DstDims:tt),*],
        $TensorTrait:tt, [$($Axes:expr),*], [$($BDims:tt),*], $DeviceTrait:tt, {$($Dims:tt),*}
    ) => {
impl<$(const $Dims: usize, )* H: Tape> $TensorTrait<$($Axes, )* $($BDims),*> for $SrcTy<$($SrcDims, )* H> {
    type Broadcasted = $DstTy<$($DstDims, )* H>;
    fn const_broadcast(self) -> Self::Broadcasted {
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

#[rustfmt::skip]
impl_broadcast!(Tensor0D, [], Tensor1D, [M], ConstBroadcast1, [0], [M], ForEachBroadcast1, {M});
#[rustfmt::skip]
impl_broadcast!(Tensor0D, [], Tensor1D, [M], ConstBroadcast1, [-1], [M], ForEachBroadcast1, {M});

impl_broadcast!(Tensor0D, [], Tensor2D, [M, N], ConstBroadcast2, [0, 1], [M, N], ForEachBroadcast2, {M, N});
impl_broadcast!(Tensor0D, [], Tensor3D, [M, N, O], ConstBroadcast3, [0, 1, 2], [M, N, O], ForEachBroadcast3, {M, N, O});
impl_broadcast!(Tensor0D, [], Tensor4D, [M, N, O, P], ConstBroadcast4, [0, 1, 2, 3], [M, N, O, P], ForEachBroadcast4, {M, N, O, P});

impl_broadcast!(Tensor1D, [M], Tensor2D, [M, N], ConstBroadcast1, [1], [N], ForEachBroadcast1, {M, N});
impl_broadcast!(Tensor1D, [M], Tensor2D, [M, N], ConstBroadcast1, [-1], [N], ForEachBroadcast1, {M, N});
impl_broadcast!(Tensor1D, [N], Tensor2D, [M, N], ConstBroadcast1, [0], [M], ForEachBroadcast1, {M, N});
impl_broadcast!(Tensor1D, [M], Tensor3D, [M, N, O], ConstBroadcast2, [1, 2], [N, O], ForEachBroadcast2, {M, N, O});
impl_broadcast!(Tensor1D, [N], Tensor3D, [M, N, O], ConstBroadcast2, [0, 2], [M, O], ForEachBroadcast2, {M, N, O});
impl_broadcast!(Tensor1D, [O], Tensor3D, [M, N, O], ConstBroadcast2, [0, 1], [M, N], ForEachBroadcast2, {M, N, O});
impl_broadcast!(Tensor1D, [M], Tensor4D, [M, N, O, P], ConstBroadcast3, [1, 2, 3], [N, O, P], ForEachBroadcast3, {M, N, O, P});
impl_broadcast!(Tensor1D, [N], Tensor4D, [M, N, O, P], ConstBroadcast3, [0, 2, 3], [M, O, P], ForEachBroadcast3, {M, N, O, P});
impl_broadcast!(Tensor1D, [O], Tensor4D, [M, N, O, P], ConstBroadcast3, [0, 1, 3], [M, N, P], ForEachBroadcast3, {M, N, O, P});
impl_broadcast!(Tensor1D, [P], Tensor4D, [M, N, O, P], ConstBroadcast3, [0, 1, 2], [M, N, O], ForEachBroadcast3, {M, N, O, P});

impl_broadcast!(Tensor2D, [M, N], Tensor3D, [M, N, O], ConstBroadcast1, [2], [O], ForEachBroadcast1, {M, N, O});
impl_broadcast!(Tensor2D, [M, N], Tensor3D, [M, N, O], ConstBroadcast1, [-1], [O], ForEachBroadcast1, {M, N, O});
impl_broadcast!(Tensor2D, [M, O], Tensor3D, [M, N, O], ConstBroadcast1, [1], [N], ForEachBroadcast1, {M, N, O});
impl_broadcast!(Tensor2D, [N, O], Tensor3D, [M, N, O], ConstBroadcast1, [0], [M], ForEachBroadcast1, {M, N, O});
impl_broadcast!(Tensor2D, [M, N], Tensor4D, [M, N, O, P], ConstBroadcast2, [2, 3], [O, P], ForEachBroadcast2, {M, N, O, P});
impl_broadcast!(Tensor2D, [M, O], Tensor4D, [M, N, O, P], ConstBroadcast2, [1, 3], [N, P], ForEachBroadcast2, {M, N, O, P});
impl_broadcast!(Tensor2D, [M, P], Tensor4D, [M, N, O, P], ConstBroadcast2, [1, 2], [N, O], ForEachBroadcast2, {M, N, O, P});
impl_broadcast!(Tensor2D, [N, O], Tensor4D, [M, N, O, P], ConstBroadcast2, [0, 3], [M, P], ForEachBroadcast2, {M, N, O, P});
impl_broadcast!(Tensor2D, [N, P], Tensor4D, [M, N, O, P], ConstBroadcast2, [0, 2], [M, O], ForEachBroadcast2, {M, N, O, P});
impl_broadcast!(Tensor2D, [O, P], Tensor4D, [M, N, O, P], ConstBroadcast2, [0, 1], [M, N], ForEachBroadcast2, {M, N, O, P});

impl_broadcast!(Tensor3D, [M, N, O], Tensor4D, [M, N, O, P], ConstBroadcast1, [3], [P], ForEachBroadcast1, {M, N, O, P});
impl_broadcast!(Tensor3D, [M, N, O], Tensor4D, [M, N, O, P], ConstBroadcast1, [-1], [P], ForEachBroadcast1, {M, N, O, P});
impl_broadcast!(Tensor3D, [M, N, P], Tensor4D, [M, N, O, P], ConstBroadcast1, [2], [O], ForEachBroadcast1, {M, N, O, P});
impl_broadcast!(Tensor3D, [M, O, P], Tensor4D, [M, N, O, P], ConstBroadcast1, [1], [N], ForEachBroadcast1, {M, N, O, P});
impl_broadcast!(Tensor3D, [N, O, P], Tensor4D, [M, N, O, P], ConstBroadcast1, [0], [M], ForEachBroadcast1, {M, N, O, P});

/// TODO
pub trait Broadcast1<const M: usize>: Sized {
    fn broadcast1<const I: isize>(self) -> Self::Broadcasted
    where
        Self: ConstBroadcast1<I, M>,
    {
        self.const_broadcast()
    }
}

/// TODO
pub trait Broadcast2<const M: usize, const N: usize>: Sized {
    fn broadcast2<const I1: isize, const I2: isize>(self) -> Self::Broadcasted
    where
        Self: ConstBroadcast2<I1, I2, M, N>,
    {
        self.const_broadcast()
    }
}

/// TODO
pub trait Broadcast3<const M: usize, const N: usize, const O: usize>: Sized {
    fn broadcast3<const I1: isize, const I2: isize, const I3: isize>(self) -> Self::Broadcasted
    where
        Self: ConstBroadcast3<I1, I2, I3, M, N, O>,
    {
        self.const_broadcast()
    }
}

/// TODO
pub trait Broadcast4<const M: usize, const N: usize, const O: usize, const P: usize>:
    Sized
{
    fn broadcast4<const I1: isize, const I2: isize, const I3: isize, const I4: isize>(
        self,
    ) -> Self::Broadcasted
    where
        Self: ConstBroadcast4<I1, I2, I3, I4, M, N, O, P>,
    {
        self.const_broadcast()
    }
}

impl<const M: usize, H: Tape> Broadcast1<M> for Tensor0D<H> {}
impl<const M: usize, const N: usize, H: Tape> Broadcast1<M> for Tensor1D<N, H> {}
impl<const M: usize, const N: usize, const O: usize, H: Tape> Broadcast1<M> for Tensor2D<N, O, H> {}
impl<const M: usize, const N: usize, const O: usize, const P: usize, H: Tape> Broadcast1<M>
    for Tensor3D<N, O, P, H>
{
}

impl<const M: usize, const N: usize, H: Tape> Broadcast2<M, N> for Tensor0D<H> {}
impl<const M: usize, const N: usize, const O: usize, H: Tape> Broadcast2<M, N> for Tensor1D<O, H> {}
impl<const M: usize, const N: usize, const O: usize, const P: usize, H: Tape> Broadcast2<M, N>
    for Tensor2D<O, P, H>
{
}

impl<const M: usize, const N: usize, const O: usize, H: Tape> Broadcast3<M, N, O> for Tensor0D<H> {}
impl<const M: usize, const N: usize, const O: usize, const P: usize, H: Tape> Broadcast3<M, N, O>
    for Tensor1D<P, H>
{
}

impl<const M: usize, const N: usize, const O: usize, const P: usize, H: Tape> Broadcast4<M, N, O, P>
    for Tensor0D<H>
{
}

/// TODO
pub trait Broadcast1To<T, const I: isize> {
    fn broadcast_to(self) -> T;
}

macro_rules! impl_broadcast_to_1 {
    ($SrcTy:tt, [$($SrcDims:tt),*], $DstTy:tt, [$($DstDims:tt),*], $Axis:expr) => {
impl<$(const $DstDims: usize, )* H: Tape> Broadcast1To<$DstTy<$($DstDims, )* H>, $Axis> for $SrcTy<$($SrcDims, )* H> {
    fn broadcast_to(self) -> $DstTy<$($DstDims, )* H> {
        self.broadcast1::<$Axis>()
    }
}
    };
}

impl<H: Tape> Broadcast1To<Tensor0D<H>, 0> for Tensor0D<H> {
    fn broadcast_to(self) -> Tensor0D<H> {
        self
    }
}
impl<H: Tape> Broadcast1To<Tensor0D<H>, -1> for Tensor0D<H> {
    fn broadcast_to(self) -> Tensor0D<H> {
        self
    }
}

impl_broadcast_to_1!(Tensor0D, [], Tensor1D, [M], -1);
impl_broadcast_to_1!(Tensor0D, [], Tensor1D, [M], 0);
impl_broadcast_to_1!(Tensor1D, [M], Tensor2D, [M, N], -1);
impl_broadcast_to_1!(Tensor1D, [M], Tensor2D, [M, N], 1);
impl_broadcast_to_1!(Tensor1D, [N], Tensor2D, [M, N], 0);
impl_broadcast_to_1!(Tensor2D, [M, N], Tensor3D, [M, N, O], -1);
impl_broadcast_to_1!(Tensor2D, [M, N], Tensor3D, [M, N, O], 2);
impl_broadcast_to_1!(Tensor2D, [M, O], Tensor3D, [M, N, O], 1);
impl_broadcast_to_1!(Tensor2D, [N, O], Tensor3D, [M, N, O], 0);
impl_broadcast_to_1!(Tensor3D, [M, N, O], Tensor4D, [M, N, O, P], -1);
impl_broadcast_to_1!(Tensor3D, [M, N, O], Tensor4D, [M, N, O, P], 3);
impl_broadcast_to_1!(Tensor3D, [M, N, P], Tensor4D, [M, N, O, P], 2);
impl_broadcast_to_1!(Tensor3D, [M, O, P], Tensor4D, [M, N, O, P], 1);
impl_broadcast_to_1!(Tensor3D, [N, O, P], Tensor4D, [M, N, O, P], 0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_1d_broadcasts() {
        let _: Tensor1D<5> = Tensor0D::zeros().broadcast1::<0>();

        let _: Tensor2D<5, 3> = Tensor1D::<3>::zeros().broadcast1::<0>();
        let _: Tensor2D<5, 3> = Tensor1D::<5>::zeros().broadcast1::<1>();
        let _: Tensor2D<5, 3> = Tensor1D::<5>::zeros().broadcast1::<-1>();

        let _: Tensor3D<3, 5, 7> = Tensor2D::<5, 7>::zeros().broadcast1::<0>();
        let _: Tensor3D<3, 5, 7> = Tensor2D::<3, 7>::zeros().broadcast1::<1>();
        let _: Tensor3D<3, 5, 7> = Tensor2D::<3, 5>::zeros().broadcast1::<2>();
        let _: Tensor3D<3, 5, 7> = Tensor2D::<3, 5>::zeros().broadcast1::<-1>();

        let _: Tensor4D<3, 5, 7, 9> = Tensor3D::<5, 7, 9>::zeros().broadcast1::<0>();
        let _: Tensor4D<3, 5, 7, 9> = Tensor3D::<3, 7, 9>::zeros().broadcast1::<1>();
        let _: Tensor4D<3, 5, 7, 9> = Tensor3D::<3, 5, 9>::zeros().broadcast1::<2>();
        let _: Tensor4D<3, 5, 7, 9> = Tensor3D::<3, 5, 7>::zeros().broadcast1::<3>();
        let _: Tensor4D<3, 5, 7, 9> = Tensor3D::<3, 5, 7>::zeros().broadcast1::<-1>();
    }

    #[test]
    fn test_valid_2d_broadcasts() {
        let _: Tensor2D<5, 3> = Tensor0D::zeros().broadcast2();

        let _: Tensor3D<3, 5, 7> = Tensor1D::<3>::zeros().broadcast2::<1, 2>();
        let _: Tensor3D<3, 5, 7> = Tensor1D::<5>::zeros().broadcast2::<0, 2>();
        let _: Tensor3D<3, 5, 7> = Tensor1D::<7>::zeros().broadcast2::<0, 1>();

        let _: Tensor4D<3, 5, 7, 9> = Tensor2D::<3, 5>::zeros().broadcast2::<2, 3>();
        let _: Tensor4D<3, 5, 7, 9> = Tensor2D::<3, 7>::zeros().broadcast2::<1, 3>();
        let _: Tensor4D<3, 5, 7, 9> = Tensor2D::<3, 9>::zeros().broadcast2::<1, 2>();
        let _: Tensor4D<3, 5, 7, 9> = Tensor2D::<5, 7>::zeros().broadcast2::<0, 3>();
        let _: Tensor4D<3, 5, 7, 9> = Tensor2D::<5, 9>::zeros().broadcast2::<0, 2>();
        let _: Tensor4D<3, 5, 7, 9> = Tensor2D::<7, 9>::zeros().broadcast2::<0, 1>();
    }

    #[test]
    fn test_valid_3d_broadcasts() {
        let _: Tensor3D<3, 5, 7> = Tensor0D::zeros().broadcast3();

        let _: Tensor4D<3, 5, 7, 9> = Tensor1D::<3>::zeros().broadcast3::<1, 2, 3>();
        let _: Tensor4D<3, 5, 7, 9> = Tensor1D::<5>::zeros().broadcast3::<0, 2, 3>();
        let _: Tensor4D<3, 5, 7, 9> = Tensor1D::<7>::zeros().broadcast3::<0, 1, 3>();
        let _: Tensor4D<3, 5, 7, 9> = Tensor1D::<9>::zeros().broadcast3::<0, 1, 2>();
    }

    #[test]
    fn test_broadcast_backwards() {
        todo!();
    }
}
