use crate::gradients::{BinaryOp, GradientTape, OpType, Operation, UnaryOp};
use crate::prelude::*;
use ndarray::prelude::{Array, Dimension};

pub(super) fn add_unary_op<Inp, Out, D>(
    tape: &mut Box<GradientTape>,
    operands: (&Inp, &Out),
    deriv: Array<f32, D>,
) where
    Inp: HasUniqueId + IsShapedArray,
    Out: HasUniqueId + IsShapedArray,
    D: Dimension,
{
    let parent_grad = tape.gradient_ref_for(operands.0.id(), operands.0.shape());
    let parent_deriv = tape.store_derivative(deriv);
    let result_grad = tape.gradient_ref_for(operands.1.id(), operands.1.shape());
    tape.add_operation(Operation::Unary(UnaryOp {
        parent_grad,
        parent_deriv,
        result_grad,
    }));
}

pub(super) fn add_binary_op<Lhs, Rhs, Out, D1, D2>(
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
