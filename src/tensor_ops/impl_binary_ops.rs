use super::ops::add_binary_op;
use crate::gradients::OpType;
use crate::prelude::*;
use ndarray::prelude::*;
use std::ops::{Add, Mul, Sub};

pub fn matmat_mul<const M: usize, const N: usize, const O: usize, H: TapeHolder>(
    lhs: Tensor2D<M, N, H>,
    rhs: &Tensor2D<N, O, NoTape>,
) -> Tensor2D<M, O, H> {
    let result = Tensor2D::new(lhs.data().dot(rhs.data()));
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    tape_holder.update_with(|tape| {
        add_binary_op(
            tape,
            OpType::MatMul { m: M, n: N, o: O },
            (&lhs, rhs, &result),
            // NOTE: the derivatives here are reversed for matrix multiplication
            rhs.data().clone(),
            lhs.data().clone(),
        )
    });
    result.with_tape_holder(tape_holder)
}

impl<const M: usize, const N: usize, const O: usize, H: TapeHolder> Mul<&Tensor2D<N, O, NoTape>>
    for Tensor2D<M, N, H>
{
    type Output = Tensor2D<M, O, H>;
    fn mul(self, rhs: &Tensor2D<N, O, NoTape>) -> Self::Output {
        matmat_mul(self, rhs)
    }
}

pub fn vecmat_mul<const N: usize, const O: usize, H: TapeHolder>(
    lhs: Tensor1D<N, H>,
    rhs: &Tensor2D<N, O, NoTape>,
) -> Tensor1D<O, H> {
    let result = Tensor1D::new(lhs.data().dot(rhs.data()));
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    tape_holder.update_with(|tape| {
        add_binary_op(
            tape,
            OpType::MatMul { m: 1, n: N, o: O },
            (&lhs, rhs, &result),
            // NOTE: the derivatives here are reversed for matrix multiplication
            rhs.data().clone(),
            lhs.data().clone(),
        )
    });
    result.with_tape_holder(tape_holder)
}

impl<const N: usize, const O: usize, H: TapeHolder> Mul<&Tensor2D<N, O, NoTape>>
    for Tensor1D<N, H>
{
    type Output = Tensor1D<O, H>;
    fn mul(self, rhs: &Tensor2D<N, O, NoTape>) -> Self::Output {
        vecmat_mul(self, rhs)
    }
}

pub fn broadcast_left_add<const M: usize, const N: usize, H: TapeHolder>(
    lhs: Tensor2D<M, N, H>,
    rhs: &Tensor1D<N, NoTape>,
) -> Tensor2D<M, N, H> {
    let result = Tensor2D::new(lhs.data() + rhs.data());
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    tape_holder.update_with(|tape| {
        add_binary_op(
            tape,
            OpType::Broadcast(ndarray::Axis(0), false),
            (&lhs, rhs, &result),
            Array::from_elem(lhs.shape(), 1.0),
            Array::from_elem(rhs.shape(), 1.0 / M as f32),
        )
    });
    result.with_tape_holder(tape_holder)
}

impl<const M: usize, const N: usize, H: TapeHolder> Add<&Tensor1D<N, NoTape>>
    for Tensor2D<M, N, H>
{
    type Output = Tensor2D<M, N, H>;
    fn add(self, rhs: &Tensor1D<N, NoTape>) -> Self::Output {
        broadcast_left_add(self, rhs)
    }
}

pub fn add<T: Tensor>(lhs: &T::NoTape, rhs: T) -> T {
    let result = T::NoTape::new(lhs.data() + rhs.data());
    let (rhs, mut tape_holder) = rhs.split_tape_holder();
    tape_holder.update_with(|tape| {
        add_binary_op(
            tape,
            OpType::Normal,
            (lhs, &rhs, &result),
            Array::from_elem(lhs.shape(), 1.0),
            Array::from_elem(rhs.shape(), 1.0),
        );
    });
    result.with_tape_holder(tape_holder)
}

pub fn sub<T: Tensor>(lhs: &T::NoTape, rhs: T) -> T {
    let result = T::NoTape::new(lhs.data() - rhs.data());
    let (rhs, mut tape_holder) = rhs.split_tape_holder();
    tape_holder.update_with(|tape| {
        add_binary_op(
            tape,
            OpType::Normal,
            (lhs, &rhs, &result),
            Array::from_elem(lhs.shape(), 1.0),
            Array::from_elem(rhs.shape(), -1.0),
        );
    });
    result.with_tape_holder(tape_holder)
}

pub fn mul<T: Tensor>(lhs: &T::NoTape, rhs: T) -> T {
    let result = T::NoTape::new(lhs.data() * rhs.data());
    let (rhs, mut tape_holder) = rhs.split_tape_holder();
    tape_holder.update_with(|tape| {
        add_binary_op(
            tape,
            OpType::Normal,
            (lhs, &rhs, &result),
            rhs.data().clone(),
            lhs.data().clone(),
        );
    });
    result.with_tape_holder(tape_holder)
}

macro_rules! binary_ops_impl {
    ($typename:ident, [$($Vs:tt),*]) => {

// &T<NoTape> + T<H>
impl<$(const $Vs: usize, )* H> Add<$typename<$($Vs, )* H>> for &$typename<$($Vs, )* NoTape>
where
    H: TapeHolder
{
    type Output = $typename<$($Vs, )* H>;
    fn add(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
        add(self, rhs)
    }
}

// &T<NoTape> - T<H>
impl<$(const $Vs: usize, )* H> Sub<$typename<$($Vs, )* H>> for &$typename<$($Vs, )* NoTape>
where
    H: TapeHolder
{
    type Output = $typename<$($Vs, )* H>;
    fn sub(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
        sub(self, rhs)
    }
}

// &T<NoTape> * T<H>
impl<$(const $Vs: usize, )* H> Mul<$typename<$($Vs, )* H>> for &$typename<$($Vs, )* NoTape>
where
    H: TapeHolder
{
    type Output = $typename<$($Vs, )* H>;
    fn mul(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
        mul(self, rhs)
    }
}
    };
}

binary_ops_impl!(Tensor0D, []);
binary_ops_impl!(Tensor1D, [N]);
binary_ops_impl!(Tensor2D, [M, N]);
binary_ops_impl!(Tensor3D, [M, N, O]);
binary_ops_impl!(Tensor4D, [M, N, O, P]);

macro_rules! broadcast_sub_impl {
    ($typename:ident, [$($Vs:tt),*], [$($Zs:tt),*], $ax:expr) => {
impl<$(const $Vs: usize, )* H: TapeHolder> std::ops::Sub<&$typename<$($Zs, )* NoTape>> for $typename<$($Vs, )* H> {
    type Output = Self;
    fn sub(self, rhs: &$typename<$($Zs, )* NoTape>) -> Self::Output {
        let result = <Self::Output as Tensor>::NoTape::new(self.data() - rhs.data());
        let (lhs, mut tape_holder) = self.split_tape_holder();
        tape_holder.update_with(|tape| {
            add_binary_op(
                tape,
                crate::gradients::OpType::Broadcast($ax, true),
                (&lhs, rhs, &result),
                Array::from_elem(lhs.shape(), 1.0),
                Array::from_elem(rhs.shape(), -1.0),
            )
        });
        result.with_tape_holder(tape_holder)
    }
}
    };
}

broadcast_sub_impl!(Tensor1D, [M], [1], Axis(0));
broadcast_sub_impl!(Tensor2D, [M, N], [M, 1], Axis(1));
broadcast_sub_impl!(Tensor3D, [M, N, O], [M, N, 1], Axis(2));
broadcast_sub_impl!(Tensor4D, [M, N, O, P], [M, N, O, 1], Axis(3));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_sub_1d() {
        let a: Tensor1D<3> = Tensor1D::new(arr1(&[1.0, 2.0, 3.0]));
        let b: Tensor1D<1> = Tensor1D::new(arr1(&[1.0]));
        let r = a.with_tape() - &b;
        assert_eq!(r.data(), arr1(&[0.0, 1.0, 2.0]));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients
                .gradient_for(a.id())
                .clone()
                .to_shape((3,))
                .unwrap(),
            arr1(&[1.0 / 3.0; 3])
        );
        assert_eq!(
            gradients
                .gradient_for(b.id())
                .clone()
                .to_shape((1,))
                .unwrap(),
            arr1(&[-1.0; 1])
        );
    }

    #[test]
    fn test_broadcast_sub_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        let b: Tensor2D<2, 1> = Tensor2D::new(arr2(&[[1.0], [2.0]]));
        // let r = broadcast_sub_2d(a.with_tape(), &b);
        let r = a.with_tape() - &b;
        assert_eq!(r.data(), arr2(&[[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients
                .gradient_for(a.id())
                .clone()
                .to_shape((2, 3))
                .unwrap(),
            arr2(&[[1.0 / 6.0; 3]; 2])
        );
        assert_eq!(
            gradients
                .gradient_for(b.id())
                .clone()
                .to_shape((2, 1))
                .unwrap(),
            arr2(&[[-0.5; 1]; 2])
        );
    }
}
