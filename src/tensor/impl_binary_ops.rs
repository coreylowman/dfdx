use super::structs::*;
use super::traits::IsShapedArray;
use super::{HasUniqueId, NoTape, TensorNoTape, TensorWithTape, WithTape};
use crate::gradients::{BinaryOp, OpType, Operation};
use crate::prelude::GradientTape;
use ndarray::prelude::Array;
use ndarray::Dimension;

fn binary_op<
    LHS: HasUniqueId + IsShapedArray,
    RHS: HasUniqueId + IsShapedArray,
    O: HasUniqueId + IsShapedArray,
    D1: Dimension,
    D2: Dimension,
>(
    mut tape: Box<GradientTape>,
    op_type: OpType,
    operands: (&LHS, &RHS, &O),
    lhs_deriv: Array<f32, D1>,
    rhs_deriv: Array<f32, D2>,
) -> Box<GradientTape> {
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
    tape
}

pub fn add_no_tape<T: TensorNoTape>(lhs: &T, rhs: T) -> T {
    T::new_no_tape(lhs.data() + rhs.data())
}

pub fn add_with_tape<T: TensorNoTape>(lhs: &T, rhs: T::WithTape) -> T::WithTape {
    let (rhs, tape) = rhs.without_tape();
    let result = T::new_no_tape(lhs.data() + rhs.data());
    let tape = binary_op(
        tape,
        OpType::Normal,
        (lhs, &rhs, &result),
        Array::from_elem(lhs.shape(), 1.0),
        Array::from_elem(rhs.shape(), 1.0),
    );
    result.put_tape(tape)
}

pub fn sub_no_tape<T: TensorNoTape>(lhs: &T, rhs: T) -> T {
    T::new_no_tape(lhs.data() - rhs.data())
}

pub fn sub_with_tape<T: TensorNoTape>(lhs: &T, rhs: T::WithTape) -> T::WithTape {
    let (rhs, tape) = rhs.without_tape();
    let result = T::new_no_tape(lhs.data() - rhs.data());
    let tape = binary_op(
        tape,
        OpType::Normal,
        (lhs, &rhs, &result),
        Array::from_elem(lhs.shape(), 1.0),
        Array::from_elem(rhs.shape(), -1.0),
    );
    result.put_tape(tape)
}

pub fn matmat_mul_no_tape<const M: usize, const N: usize, const O: usize>(
    lhs: Tensor2D<M, N, NoTape>,
    rhs: &Tensor2D<N, O, NoTape>,
) -> Tensor2D<M, O, NoTape> {
    Tensor2D::new_no_tape(lhs.data().dot(rhs.data()))
}

impl<const M: usize, const N: usize, const O: usize> std::ops::Mul<&Tensor2D<N, O, NoTape>>
    for Tensor2D<M, N, NoTape>
{
    type Output = Tensor2D<M, O, NoTape>;
    fn mul(self, rhs: &Tensor2D<N, O, NoTape>) -> Self::Output {
        matmat_mul_no_tape(self, rhs)
    }
}

pub fn matmat_mul_with_tape<const M: usize, const N: usize, const O: usize>(
    lhs: Tensor2D<M, N, WithTape>,
    rhs: &Tensor2D<N, O, NoTape>,
) -> Tensor2D<M, O, WithTape> {
    let (lhs, tape) = lhs.without_tape();
    let result = Tensor2D::new_no_tape(lhs.data().dot(rhs.data()));
    let tape = binary_op(
        tape,
        OpType::MatMul { m: M, n: N, o: O },
        (&lhs, rhs, &result),
        // NOTE: the derivatives here are reversed for matrix multiplication
        rhs.data.clone(),
        lhs.data.clone(),
    );
    result.put_tape(tape)
}

impl<const M: usize, const N: usize, const O: usize> std::ops::Mul<&Tensor2D<N, O, NoTape>>
    for Tensor2D<M, N, WithTape>
{
    type Output = Tensor2D<M, O, WithTape>;
    fn mul(self, rhs: &Tensor2D<N, O, NoTape>) -> Self::Output {
        matmat_mul_with_tape(self, rhs)
    }
}

pub fn vecmat_mul_no_tape<const N: usize, const O: usize>(
    lhs: Tensor1D<N, NoTape>,
    rhs: &Tensor2D<N, O, NoTape>,
) -> Tensor1D<O, NoTape> {
    Tensor1D::new_no_tape(lhs.data().dot(rhs.data()))
}

impl<const N: usize, const O: usize> std::ops::Mul<&Tensor2D<N, O, NoTape>>
    for Tensor1D<N, NoTape>
{
    type Output = Tensor1D<O, NoTape>;
    fn mul(self, rhs: &Tensor2D<N, O, NoTape>) -> Self::Output {
        vecmat_mul_no_tape(self, rhs)
    }
}

pub fn vecmat_mul_with_tape<const N: usize, const O: usize>(
    lhs: Tensor1D<N, WithTape>,
    rhs: &Tensor2D<N, O, NoTape>,
) -> Tensor1D<O, WithTape> {
    let (lhs, tape) = lhs.without_tape();
    let result = Tensor1D::new_no_tape(lhs.data().dot(rhs.data()));
    let tape = binary_op(
        tape,
        OpType::MatMul { m: 1, n: N, o: O },
        (&lhs, rhs, &result),
        // NOTE: the derivatives here are reversed for matrix multiplication
        rhs.data.clone(),
        lhs.data.clone(),
    );
    result.put_tape(tape)
}

impl<const N: usize, const O: usize> std::ops::Mul<&Tensor2D<N, O, NoTape>>
    for Tensor1D<N, WithTape>
{
    type Output = Tensor1D<O, WithTape>;
    fn mul(self, rhs: &Tensor2D<N, O, NoTape>) -> Self::Output {
        vecmat_mul_with_tape(self, rhs)
    }
}
pub fn broadcast_add_no_tape<const M: usize, const N: usize>(
    lhs: Tensor2D<M, N, NoTape>,
    rhs: &Tensor1D<N, NoTape>,
) -> Tensor2D<M, N, NoTape> {
    Tensor2D::new_no_tape(lhs.data() + rhs.data())
}

impl<const M: usize, const N: usize> std::ops::Add<&Tensor1D<N, NoTape>>
    for Tensor2D<M, N, NoTape>
{
    type Output = Tensor2D<M, N, NoTape>;
    fn add(self, rhs: &Tensor1D<N, NoTape>) -> Self::Output {
        broadcast_add_no_tape(self, rhs)
    }
}

pub fn broadcast_add_with_tape<const M: usize, const N: usize>(
    lhs: Tensor2D<M, N, WithTape>,
    rhs: &Tensor1D<N, NoTape>,
) -> Tensor2D<M, N, WithTape> {
    let (lhs, tape) = lhs.without_tape();
    let result = Tensor2D::new_no_tape(lhs.data() + rhs.data());
    let tape = binary_op(
        tape,
        OpType::Broadcast,
        (&lhs, rhs, &result),
        Array::from_elem(lhs.shape(), 1.0),
        Array::from_elem(rhs.shape(), 1.0 / M as f32),
    );
    result.put_tape(tape)
}

impl<const M: usize, const N: usize> std::ops::Add<&Tensor1D<N, NoTape>>
    for Tensor2D<M, N, WithTape>
{
    type Output = Tensor2D<M, N, WithTape>;
    fn add(self, rhs: &Tensor1D<N, NoTape>) -> Self::Output {
        broadcast_add_with_tape(self, rhs)
    }
}

macro_rules! binary_ops_impl {
    ($typename:ident, [$($const_names:tt),*]) => {

// &T<NoTape> + T<NoTape>
impl<$(const $const_names: usize),*> std::ops::Add<$typename<$($const_names, )* NoTape>> for &$typename<$($const_names, )* NoTape> {
    type Output = $typename<$($const_names, )* NoTape>;
    fn add(self, rhs: $typename<$($const_names, )* NoTape>) -> Self::Output {
        add_no_tape(self, rhs)
    }
}

// &T<NoTape> + T<WithTape>
impl<$(const $const_names: usize),*> std::ops::Add<$typename<$($const_names, )* WithTape>> for &$typename<$($const_names, )* NoTape> {
    type Output = $typename<$($const_names, )* WithTape>;
    fn add(self, rhs: $typename<$($const_names, )* WithTape>) -> Self::Output {
        add_with_tape(self, rhs)
    }
}

// &T<NoTape> - T<NoTape>
impl<$(const $const_names: usize),*> std::ops::Sub<$typename<$($const_names, )* NoTape>> for &$typename<$($const_names, )* NoTape> {
    type Output = $typename<$($const_names, )* NoTape>;
    fn sub(self, rhs: $typename<$($const_names, )* NoTape>) -> Self::Output {
        sub_no_tape(self, rhs)
    }
}

// &T<NoTape> - T<WithTape>
impl<$(const $const_names: usize),*> std::ops::Sub<$typename<$($const_names, )* WithTape>> for &$typename<$($const_names, )* NoTape> {
    type Output = $typename<$($const_names, )* WithTape>;
    fn sub(self, rhs: $typename<$($const_names, )* WithTape>) -> Self::Output {
        sub_with_tape(self, rhs)
    }
}
    };
}

binary_ops_impl!(Tensor0D, []);
binary_ops_impl!(Tensor1D, [N]);
binary_ops_impl!(Tensor2D, [M, N]);
binary_ops_impl!(Tensor3D, [M, N, O]);
binary_ops_impl!(Tensor4D, [M, N, O, P]);
