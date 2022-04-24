use std::collections::HashMap;

#[derive(Debug)]
pub struct GradientTape {
    pub(super) grad_ref_by_id: HashMap<usize, GradientRef>,
    pub(super) operations: Vec<Operation>,
    pub(super) derivatives: Vec<ndarray::ArrayD<f32>>,
    pub(super) gradients: Vec<ndarray::ArrayD<f32>>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Operation {
    Unary(UnaryOp),
    Binary(BinaryOp),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct UnaryOp {
    pub(crate) parent_grad: GradientRef,
    pub(crate) parent_deriv: DerivativeRef,
    pub(crate) result_grad: GradientRef,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BinaryOp {
    pub(crate) op_type: OpType,
    pub(crate) parent_grads: [GradientRef; 2],
    pub(crate) parent_derivs: [DerivativeRef; 2],
    pub(crate) result_grad: GradientRef,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum OpType {
    Normal,
    Broadcast(ndarray::Axis, bool),
    MatMul { m: usize, n: usize, o: usize },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DerivativeRef {
    pub(super) index: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct GradientRef {
    pub(super) index: usize,
}
