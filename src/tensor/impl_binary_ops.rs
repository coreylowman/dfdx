use super::structs::{Tensor1D, Tensor2D};
use super::traits::{HasGradientRef, HasGradientTape, IsShapedArray, OnGradientTape, Tensor};
use crate::gradients::{BinaryOp, OpType, Operation};
use ndarray::prelude::Array;

pub fn add<T>(lhs: &mut T, rhs: &mut T) -> T
where
    T: Tensor,
{
    let mut opt_tape = lhs.mut_tape().take().or(rhs.mut_tape().take());
    let opt_grad = opt_tape.as_mut().map(|mut tape| {
        lhs.put_on(&mut tape);
        rhs.put_on(&mut tape);

        let lhs_deriv = tape.store_derivative(Array::from_elem(T::SHAPE, 1.0));
        let rhs_deriv = tape.store_derivative(Array::from_elem(T::SHAPE, 1.0));
        let result_grad = tape.allocate_gradient(T::SHAPE);

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::Normal,
            parent_grads: [lhs.grad_ref().unwrap(), rhs.grad_ref().unwrap()],
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        result_grad
    });

    T::new(lhs.data() + rhs.data(), opt_grad, opt_tape)
}

pub fn sub<T>(lhs: &mut T, rhs: &mut T) -> T
where
    T: Tensor,
{
    let mut opt_tape = lhs.mut_tape().take().or(rhs.mut_tape().take());
    let opt_grad = opt_tape.as_mut().map(|mut tape| {
        lhs.put_on(&mut tape);
        rhs.put_on(&mut tape);

        let lhs_deriv = tape.store_derivative(Array::from_elem(T::SHAPE, 1.0));
        let rhs_deriv = tape.store_derivative(Array::from_elem(T::SHAPE, -1.0));
        let result_grad = tape.allocate_gradient(T::SHAPE);

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::Normal,
            parent_grads: [lhs.grad_ref().unwrap(), rhs.grad_ref().unwrap()],
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        result_grad
    });

    T::new(lhs.data() - rhs.data(), opt_grad, opt_tape)
}

pub fn matmat_mul<const M: usize, const N: usize, const O: usize>(
    lhs: &mut Tensor2D<M, N>,
    rhs: &mut Tensor2D<N, O>,
) -> Tensor2D<M, O> {
    let mut opt_tape = lhs.mut_tape().take().or(rhs.mut_tape().take());
    let opt_grad = opt_tape.as_mut().map(|mut tape| {
        lhs.put_on(&mut tape);
        rhs.put_on(&mut tape);

        let lhs_deriv = tape.store_derivative(rhs.data.clone());
        let rhs_deriv = tape.store_derivative(lhs.data.clone());
        let result_grad = tape.allocate_gradient((M, O));

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::MatMul { m: M, n: N, o: O },
            parent_grads: [lhs.grad_ref().unwrap(), rhs.grad_ref().unwrap()],
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        result_grad
    });

    Tensor2D::new(lhs.data().dot(rhs.data()), opt_grad, opt_tape)
}

pub fn vecmat_mul<const N: usize, const O: usize>(
    lhs: &mut Tensor1D<N>,
    rhs: &mut Tensor2D<N, O>,
) -> Tensor1D<O> {
    let mut opt_tape = lhs.mut_tape().take().or(rhs.mut_tape().take());
    let opt_grad = opt_tape.as_mut().map(|mut tape| {
        lhs.put_on(&mut tape);
        rhs.put_on(&mut tape);

        let lhs_deriv = tape.store_derivative(rhs.data.clone());
        let rhs_deriv = tape.store_derivative(lhs.data.clone());
        let result_grad = tape.allocate_gradient((O,));

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::MatMul { m: 1, n: N, o: O },
            parent_grads: [lhs.grad_ref().unwrap(), rhs.grad_ref().unwrap()],
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        result_grad
    });

    Tensor1D::new(lhs.data().dot(rhs.data()), opt_grad, opt_tape)
}

pub fn broadcast_add<const M: usize, const N: usize>(
    lhs: &mut Tensor2D<M, N>,
    rhs: &mut Tensor1D<N>,
) -> Tensor2D<M, N> {
    let mut opt_tape = lhs.mut_tape().take().or(rhs.mut_tape().take());
    let opt_grad = opt_tape.as_mut().map(|mut tape| {
        lhs.put_on(&mut tape);
        rhs.put_on(&mut tape);

        let lhs_deriv = tape.store_derivative(Array::from_elem((M, N), 1.0));
        let rhs_deriv = tape.store_derivative(Array::from_elem((N,), 1.0 / M as f32));
        let result_grad = tape.allocate_gradient((M, N));

        tape.add_operation(Operation::Binary(BinaryOp {
            op_type: OpType::Broadcast,
            parent_grads: [lhs.grad_ref().unwrap(), rhs.grad_ref().unwrap()],
            parent_derivs: [lhs_deriv, rhs_deriv],
            result_grad,
        }));

        result_grad
    });
    Tensor2D::new(lhs.data() + rhs.data(), opt_grad, opt_tape)
}
