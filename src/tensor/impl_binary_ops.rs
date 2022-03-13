use super::structs::{Tensor1D, Tensor2D};
use super::traits::{CanStoreGradientTape, HasUniqueId, IsShapedArray, Tensor};
use crate::gradients::{BinaryOp, OpType, Operation};
use ndarray::prelude::Array;

pub fn add<T>(lhs: &T, rhs: &T) -> T
where
    T: Tensor,
{
    let result = T::new(lhs.data() + rhs.data());
    result.put_tape(lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        let parent_grads = [
            tape.gradient_ref_for(lhs.id(), lhs.shape()),
            tape.gradient_ref_for(rhs.id(), rhs.shape()),
        ];

        let lhs_deriv = tape.store_derivative(Array::from_elem(lhs.shape(), 1.0));
        let rhs_deriv = tape.store_derivative(Array::from_elem(rhs.shape(), 1.0));
        let result_grad = tape.gradient_ref_for(result.id(), result.shape());

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::Normal,
            parent_grads,
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        tape
    }));
    result
}

pub fn sub<T>(lhs: &T, rhs: &T) -> T
where
    T: Tensor,
{
    let result = T::new(lhs.data() - rhs.data());
    result.put_tape(lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        let parent_grads = [
            tape.gradient_ref_for(lhs.id(), lhs.shape()),
            tape.gradient_ref_for(rhs.id(), rhs.shape()),
        ];

        let lhs_deriv = tape.store_derivative(Array::from_elem(lhs.shape(), 1.0));
        let rhs_deriv = tape.store_derivative(Array::from_elem(rhs.shape(), -1.0));
        let result_grad = tape.gradient_ref_for(result.id(), result.shape());

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::Normal,
            parent_grads,
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        tape
    }));
    result
}

pub fn matmat_mul<const M: usize, const N: usize, const O: usize>(
    lhs: &Tensor2D<M, N>,
    rhs: &Tensor2D<N, O>,
) -> Tensor2D<M, O> {
    let result = Tensor2D::new(lhs.data().dot(rhs.data()));
    result.put_tape(lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        let parent_grads = [
            tape.gradient_ref_for(lhs.id(), lhs.shape()),
            tape.gradient_ref_for(rhs.id(), rhs.shape()),
        ];

        let lhs_deriv = tape.store_derivative(rhs.data.clone());
        let rhs_deriv = tape.store_derivative(lhs.data.clone());
        let result_grad = tape.gradient_ref_for(result.id(), result.shape());

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::MatMul { m: M, n: N, o: O },
            parent_grads,
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        tape
    }));
    result
}

pub fn vecmat_mul<const N: usize, const O: usize>(
    lhs: &Tensor1D<N>,
    rhs: &Tensor2D<N, O>,
) -> Tensor1D<O> {
    let result = Tensor1D::new(lhs.data().dot(rhs.data()));
    result.put_tape(lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        let parent_grads = [
            tape.gradient_ref_for(lhs.id(), lhs.shape()),
            tape.gradient_ref_for(rhs.id(), rhs.shape()),
        ];

        let lhs_deriv = tape.store_derivative(rhs.data.clone());
        let rhs_deriv = tape.store_derivative(lhs.data.clone());
        let result_grad = tape.gradient_ref_for(result.id(), result.shape());

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::MatMul { m: 1, n: N, o: O },
            parent_grads,
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        tape
    }));
    result
}

pub fn broadcast_add<const M: usize, const N: usize>(
    lhs: &Tensor2D<M, N>,
    rhs: &Tensor1D<N>,
) -> Tensor2D<M, N> {
    let result = Tensor2D::new(lhs.data() + rhs.data());
    result.put_tape(lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        let parent_grads = [
            tape.gradient_ref_for(lhs.id(), lhs.shape()),
            tape.gradient_ref_for(rhs.id(), rhs.shape()),
        ];

        let lhs_deriv = tape.store_derivative(Array::from_elem((M, N), 1.0));
        let rhs_deriv = tape.store_derivative(Array::from_elem((N,), 1.0 / M as f32));
        let result_grad = tape.gradient_ref_for(result.id(), result.shape());

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::Broadcast,
            parent_grads,
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        tape
    }));
    result
}
