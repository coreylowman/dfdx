use super::structs::{Tensor1D, Tensor2D};
use super::traits::{IsShapedArray, Tensor};
use crate::gradients::{BinaryOp, HasGradient, OpType, Operation};
use ndarray::prelude::Array;

pub fn add<T>(lhs: &mut T, rhs: &mut T) -> T
where
    T: Tensor,
{
    let grad = lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        lhs.ensure_gradient_allocated(&mut tape);
        rhs.ensure_gradient_allocated(&mut tape);

        let lhs_deriv = tape.store_derivative(Array::from_elem(T::SHAPE, 1.0));
        let rhs_deriv = tape.store_derivative(Array::from_elem(T::SHAPE, 1.0));
        let mut gradient = tape.allocate_gradient(T::SHAPE);

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::Normal,
            parent_grads: [lhs.gradient_ref(), rhs.gradient_ref()],
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad: gradient.gradient_ref,
        }));

        gradient.tape = Some(tape);
        gradient
    });

    T::new(lhs.data() + rhs.data(), grad)
}

pub fn sub<T>(lhs: &mut T, rhs: &mut T) -> T
where
    T: Tensor,
{
    let grad = lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        lhs.ensure_gradient_allocated(&mut tape);
        rhs.ensure_gradient_allocated(&mut tape);

        let lhs_deriv = tape.store_derivative(Array::from_elem(T::SHAPE, 1.0));
        let rhs_deriv = tape.store_derivative(Array::from_elem(T::SHAPE, -1.0));
        let mut gradient = tape.allocate_gradient(T::SHAPE);

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::Normal,
            parent_grads: [lhs.gradient_ref(), rhs.gradient_ref()],
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad: gradient.gradient_ref,
        }));

        gradient.tape = Some(tape);
        gradient
    });

    T::new(lhs.data() - rhs.data(), grad)
}

pub fn matmat_mul<const M: usize, const N: usize, const O: usize>(
    lhs: &mut Tensor2D<M, N>,
    rhs: &mut Tensor2D<N, O>,
) -> Tensor2D<M, O> {
    let grad = lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        lhs.ensure_gradient_allocated(&mut tape);
        rhs.ensure_gradient_allocated(&mut tape);

        let lhs_deriv = tape.store_derivative(rhs.data.clone());
        let rhs_deriv = tape.store_derivative(lhs.data.clone());
        let mut gradient = tape.allocate_gradient((M, O));

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::MatMul { m: M, n: N, o: O },
            parent_grads: [lhs.gradient_ref(), rhs.gradient_ref()],
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad: gradient.gradient_ref,
        }));

        gradient.tape = Some(tape);
        gradient
    });

    Tensor2D::new(lhs.data().dot(rhs.data()), grad)
}

pub fn vecmat_mul<const N: usize, const O: usize>(
    lhs: &mut Tensor1D<N>,
    rhs: &mut Tensor2D<N, O>,
) -> Tensor1D<O> {
    let grad = lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        lhs.ensure_gradient_allocated(&mut tape);
        rhs.ensure_gradient_allocated(&mut tape);

        let lhs_deriv = tape.store_derivative(rhs.data.clone());
        let rhs_deriv = tape.store_derivative(lhs.data.clone());
        let mut gradient = tape.allocate_gradient((O,));

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::MatMul { m: 1, n: N, o: O },
            parent_grads: [lhs.gradient_ref(), rhs.gradient_ref()],
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad: gradient.gradient_ref,
        }));

        gradient.tape = Some(tape);
        gradient
    });

    Tensor1D::new(lhs.data().dot(rhs.data()), grad)
}

pub fn broadcast_add<const M: usize, const N: usize>(
    lhs: &mut Tensor2D<M, N>,
    rhs: &mut Tensor1D<N>,
) -> Tensor2D<M, N> {
    let grad = lhs.take_tape().or(rhs.take_tape()).map(|mut tape| {
        lhs.ensure_gradient_allocated(&mut tape);
        rhs.ensure_gradient_allocated(&mut tape);

        let lhs_deriv = tape.store_derivative(Array::from_elem((M, N), 1.0));
        let rhs_deriv = tape.store_derivative(Array::from_elem((N,), 1.0 / M as f32));
        let mut gradient = tape.allocate_gradient((M, N));

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::Broadcast,
            parent_grads: [lhs.gradient_ref(), rhs.gradient_ref()],
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad: gradient.gradient_ref,
        }));

        gradient.tape = Some(tape);
        gradient
    });
    Tensor2D::new(lhs.data() + rhs.data(), grad)
}
