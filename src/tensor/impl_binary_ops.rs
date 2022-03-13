use super::structs::{Tensor1D, Tensor2D};
use super::traits::{CanStoreGradientTape, IsShapedArray, Tensor};
use crate::gradients::{BinaryOp, OpType, Operation};
use crate::prelude::GradientTape;
use ndarray::prelude::Array;
use ndarray::Dimension;

fn binary_op<LHS: Tensor, RHS: Tensor, O: Tensor, D1: Dimension, D2: Dimension>(
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

pub fn add<T>(lhs: &T, rhs: &T) -> T
where
    T: Tensor,
{
    let result = T::new(lhs.data() + rhs.data());
    result.put_tape(lhs.take_tape().or(rhs.take_tape()).map(|tape| {
        binary_op(
            tape,
            OpType::Normal,
            (lhs, rhs, &result),
            Array::from_elem(lhs.shape(), 1.0),
            Array::from_elem(rhs.shape(), 1.0),
        )
    }));
    result
}

pub fn sub<T>(lhs: &T, rhs: &T) -> T
where
    T: Tensor,
{
    let result = T::new(lhs.data() - rhs.data());
    result.put_tape(lhs.take_tape().or(rhs.take_tape()).map(|tape| {
        binary_op(
            tape,
            OpType::Normal,
            (lhs, rhs, &result),
            Array::from_elem(lhs.shape(), 1.0),
            Array::from_elem(rhs.shape(), -1.0),
        )
    }));
    result
}

pub fn matmat_mul<const M: usize, const N: usize, const O: usize>(
    lhs: &Tensor2D<M, N>,
    rhs: &Tensor2D<N, O>,
) -> Tensor2D<M, O> {
    let result = Tensor2D::new(lhs.data().dot(rhs.data()));
    result.put_tape(lhs.take_tape().or(rhs.take_tape()).map(|tape| {
        binary_op(
            tape,
            OpType::MatMul { m: M, n: N, o: O },
            (lhs, rhs, &result),
            // NOTE: the derivatives here are reversed for matrix multiplication
            rhs.data.clone(),
            lhs.data.clone(),
        )
    }));
    result
}

pub fn vecmat_mul<const N: usize, const O: usize>(
    lhs: &Tensor1D<N>,
    rhs: &Tensor2D<N, O>,
) -> Tensor1D<O> {
    let result = Tensor1D::new(lhs.data().dot(rhs.data()));
    result.put_tape(lhs.take_tape().or(rhs.take_tape()).map(|tape| {
        binary_op(
            tape,
            OpType::MatMul { m: 1, n: N, o: O },
            (lhs, rhs, &result),
            // NOTE: the derivatives here are reversed for matrix multiplication
            rhs.data.clone(),
            lhs.data.clone(),
        )
    }));
    result
}

pub fn broadcast_add<const M: usize, const N: usize>(
    lhs: &Tensor2D<M, N>,
    rhs: &Tensor1D<N>,
) -> Tensor2D<M, N> {
    let result = Tensor2D::new(lhs.data() + rhs.data());
    result.put_tape(lhs.take_tape().or(rhs.take_tape()).map(|tape| {
        binary_op(
            tape,
            OpType::Broadcast,
            (lhs, rhs, &result),
            Array::from_elem(lhs.shape(), 1.0),
            Array::from_elem(rhs.shape(), 1.0 / M as f32),
        )
    }));
    result
}
