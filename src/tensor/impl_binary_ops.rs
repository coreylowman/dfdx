use super::*;
use crate::gradients::{BinaryOp, GradientTape, OpType, Operation};
use ndarray::prelude::{Array, Dimension};
use std::ops::{Add, Mul, Sub};

fn add_binary_op<Lhs, Rhs, Out, D1, D2>(
    tape: &mut Box<GradientTape>,
    op_type: OpType,
    operands: (&Lhs, &Rhs, &Out),
    lhs_deriv: Array<f32, D1>,
    rhs_deriv: Array<f32, D2>,
) where
    Lhs: HasUniqueId + IsShapedArray,
    Rhs: HasUniqueId + IsShapedArray,
    Out: HasUniqueId + IsShapedArray,
    D1: Dimension,
    D2: Dimension,
{
    let parent_grads = [
        tape.gradient_ref_for(operands.0.id(), operands.0.shape()),
        tape.gradient_ref_for(operands.1.id(), operands.1.shape()),
    ];
    let parent_derivs = [
        tape.store_derivative(lhs_deriv),
        tape.store_derivative(rhs_deriv),
    ];
    let result_grad = tape.gradient_ref_for(operands.2.id(), operands.2.shape());
    tape.add_operation(Operation::Binary(BinaryOp {
        op_type,
        parent_grads,
        parent_derivs,
        result_grad,
    }));
}

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
            rhs.data.clone(),
            lhs.data.clone(),
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
            rhs.data.clone(),
            lhs.data.clone(),
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

pub fn broadcast_add<const M: usize, const N: usize, H: TapeHolder>(
    lhs: Tensor2D<M, N, H>,
    rhs: &Tensor1D<N, NoTape>,
) -> Tensor2D<M, N, H> {
    let result = Tensor2D::new(lhs.data() + rhs.data());
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    tape_holder.update_with(|tape| {
        add_binary_op(
            tape,
            OpType::Broadcast,
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
        broadcast_add(self, rhs)
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
