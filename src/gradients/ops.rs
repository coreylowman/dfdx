#[derive(Debug, Clone, Copy)]
pub(crate) struct DerivativeRef {
    pub(super) index: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct GradientRef {
    pub(super) index: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum OpType {
    Normal,
    Broadcast,
    MatMul { m: usize, n: usize, o: usize },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct UnaryOp {
    pub(crate) op_type: OpType,
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
pub(crate) enum Operation {
    Unary(UnaryOp),
    Binary(BinaryOp),
}
