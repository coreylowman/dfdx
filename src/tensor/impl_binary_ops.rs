use super::structs::{GradientData, Tensor1D, Tensor2D};
use super::traits::{HasGradientData, IsShapedArray, Tensor};
use crate::gradients::{BinaryOp, OpType, Operation};
use ndarray::prelude::Array;

pub fn add<T>(lhs: &T, rhs: &T) -> T
where
    T: Tensor,
{
    let grad_data = lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        let parent_grads = [lhs.grad_ref(&mut tape), rhs.grad_ref(&mut tape)];

        let lhs_deriv = tape.store_derivative(Array::from_elem(T::SHAPE, 1.0));
        let rhs_deriv = tape.store_derivative(Array::from_elem(T::SHAPE, 1.0));
        let result_grad = tape.allocate_gradient(T::SHAPE);

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::Normal,
            parent_grads,
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        GradientData {
            grad_ref: Some(result_grad),
            tape: Some(tape),
        }
    });

    T::new(
        lhs.data() + rhs.data(),
        grad_data.unwrap_or(GradientData::default()),
    )
}

pub fn sub<T>(lhs: &T, rhs: &T) -> T
where
    T: Tensor,
{
    let grad = lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        let parent_grads = [lhs.grad_ref(&mut tape), rhs.grad_ref(&mut tape)];

        let lhs_deriv = tape.store_derivative(Array::from_elem(T::SHAPE, 1.0));
        let rhs_deriv = tape.store_derivative(Array::from_elem(T::SHAPE, -1.0));
        let result_grad = tape.allocate_gradient(T::SHAPE);

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::Normal,
            parent_grads,
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        GradientData {
            grad_ref: Some(result_grad),
            tape: Some(tape),
        }
    });

    T::new(
        lhs.data() - rhs.data(),
        grad.unwrap_or(GradientData::default()),
    )
}

pub fn matmat_mul<const M: usize, const N: usize, const O: usize>(
    lhs: &Tensor2D<M, N>,
    rhs: &Tensor2D<N, O>,
) -> Tensor2D<M, O> {
    let grad = lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        let parent_grads = [lhs.grad_ref(&mut tape), rhs.grad_ref(&mut tape)];

        let lhs_deriv = tape.store_derivative(rhs.data.clone());
        let rhs_deriv = tape.store_derivative(lhs.data.clone());
        let result_grad = tape.allocate_gradient((M, O));

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::MatMul { m: M, n: N, o: O },
            parent_grads,
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        GradientData {
            grad_ref: Some(result_grad),
            tape: Some(tape),
        }
    });

    Tensor2D::new(
        lhs.data().dot(rhs.data()),
        grad.unwrap_or(GradientData::default()),
    )
}

pub fn vecmat_mul<const N: usize, const O: usize>(
    lhs: &Tensor1D<N>,
    rhs: &Tensor2D<N, O>,
) -> Tensor1D<O> {
    let grad = lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        let parent_grads = [lhs.grad_ref(&mut tape), rhs.grad_ref(&mut tape)];

        let lhs_deriv = tape.store_derivative(rhs.data.clone());
        let rhs_deriv = tape.store_derivative(lhs.data.clone());
        let result_grad = tape.allocate_gradient((O,));

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::MatMul { m: 1, n: N, o: O },
            parent_grads,
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        GradientData {
            grad_ref: Some(result_grad),
            tape: Some(tape),
        }
    });

    Tensor1D::new(
        lhs.data().dot(rhs.data()),
        grad.unwrap_or(GradientData::default()),
    )
}

pub fn broadcast_add<const M: usize, const N: usize>(
    lhs: &Tensor2D<M, N>,
    rhs: &Tensor1D<N>,
) -> Tensor2D<M, N> {
    let grad = lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        let parent_grads = [lhs.grad_ref(&mut tape), rhs.grad_ref(&mut tape)];

        let lhs_deriv = tape.store_derivative(Array::from_elem((M, N), 1.0));
        let rhs_deriv = tape.store_derivative(Array::from_elem((N,), 1.0 / M as f32));
        let result_grad = tape.allocate_gradient((M, N));

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::Broadcast,
            parent_grads,
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        GradientData {
            grad_ref: Some(result_grad),
            tape: Some(tape),
        }
    });
    Tensor2D::new(
        lhs.data() + rhs.data(),
        grad.unwrap_or(GradientData::default()),
    )
}
