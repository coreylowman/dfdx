use std::ops::Index;

use ndarray::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct DerivativeRef {
    index: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct GradientRef {
    index: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum OpType {
    Add,
    BroadcastAdd,
    Sub,
    BroadcastSub,
    MatVec { m: usize, n: usize },
    MatMul { m: usize, n: usize, o: usize },
    Square,
    Mean,
    ReLU,
}

#[derive(Debug, Clone, Copy)]
pub struct UnaryOp {
    pub op_type: OpType,
    pub parent_grad: GradientRef,
    pub parent_deriv: DerivativeRef,
    pub result_grad: GradientRef,
}

#[derive(Debug, Clone, Copy)]
pub struct BinaryOp {
    pub op_type: OpType,
    pub parent_grads: [GradientRef; 2],
    pub parent_derivs: [DerivativeRef; 2],
    pub result_grad: GradientRef,
}

#[derive(Debug, Clone, Copy)]
pub enum Operation {
    Unary(UnaryOp),
    Binary(BinaryOp),
}

#[derive(Debug)]
pub struct GradientTape {
    operations: Vec<Operation>,
    derivatives: Vec<ArrayD<f32>>,
    gradients: Vec<ArrayD<f32>>,
}

impl GradientTape {
    pub fn new() -> Self {
        Self {
            derivatives: Vec::new(),
            gradients: Vec::new(),
            operations: Vec::new(),
        }
    }

    pub fn scale(&mut self, scalar: f32) {
        for g in self.gradients.iter_mut() {
            *g *= scalar;
        }
    }

    pub fn store_derivative<D: Dimension>(&mut self, deriv: Array<f32, D>) -> DerivativeRef {
        let index = self.derivatives.len();
        self.derivatives.push(deriv.into_dyn());
        DerivativeRef { index }
    }

    pub fn store_gradient<D: Dimension, Sh: ShapeBuilder<Dim = D>>(
        &mut self,
        shape: Sh,
    ) -> GradientRef {
        let index = self.gradients.len();
        self.gradients.push(Array::zeros(shape).into_dyn());
        GradientRef { index }
    }

    pub fn add_operation(&mut self, operation: Operation) {
        self.operations.push(operation);
    }

    fn grad(&self, gradient_ref: GradientRef) -> &ArrayD<f32> {
        &self.gradients[gradient_ref.index]
    }

    fn deriv(&self, derivative_ref: DerivativeRef) -> &ArrayD<f32> {
        &self.derivatives[derivative_ref.index]
    }

    pub fn backward(&mut self, gradient_ref: GradientRef) {
        self.gradients[gradient_ref.index].fill(1.0);
        for operation in self.operations.iter().rev() {
            match operation {
                Operation::Unary(op) => {
                    let d_grad = self.deriv(op.parent_deriv) * self.grad(op.result_grad);
                    self.gradients[op.parent_grad.index] += &d_grad;
                }
                Operation::Binary(op) => match op.op_type {
                    OpType::MatVec { m, n } => {
                        let result = self
                            .grad(op.result_grad)
                            .clone()
                            .into_shape((m, 1))
                            .unwrap();

                        let wt = self
                            .deriv(op.parent_derivs[1])
                            .clone()
                            .into_shape((m, n))
                            .unwrap()
                            .reversed_axes();

                        let d_grad0 = &result * self.deriv(op.parent_derivs[0]);
                        let d_grad1 = wt.dot(&result).into_shape((n,)).unwrap();

                        self.gradients[op.parent_grads[0].index] += &d_grad0;
                        self.gradients[op.parent_grads[1].index] += &d_grad1;
                    }
                    OpType::MatMul { m, n, o } => {
                        let result = self
                            .grad(op.result_grad)
                            .clone()
                            .into_shape((m, o))
                            .unwrap();
                        let d0 = self
                            .deriv(op.parent_derivs[0])
                            .clone()
                            .into_shape((n, o))
                            .unwrap()
                            .reversed_axes();
                        let d1 = self
                            .deriv(op.parent_derivs[1])
                            .clone()
                            .into_shape((m, n))
                            .unwrap()
                            .reversed_axes();

                        let d_grad0 = result.dot(&d0);
                        let d_grad1 = d1.dot(&result);

                        self.gradients[op.parent_grads[0].index] += &d_grad0;
                        self.gradients[op.parent_grads[1].index] += &d_grad1;
                    }
                    OpType::BroadcastAdd | OpType::BroadcastSub => {
                        let d_grad = self.deriv(op.parent_derivs[0]) * self.grad(op.result_grad);
                        self.gradients[op.parent_grads[0].index] += &d_grad;

                        let d_grad = self.deriv(op.parent_derivs[1])
                            * &self.grad(op.result_grad).sum_axis(Axis(0));
                        self.gradients[op.parent_grads[1].index] += &d_grad;
                    }
                    _ => {
                        for (&deriv, grad) in op.parent_derivs.iter().zip(op.parent_grads.iter()) {
                            let d_grad = self.deriv(deriv) * self.grad(op.result_grad);
                            self.gradients[grad.index] += &d_grad;
                        }
                    }
                },
            }
        }
    }
}

impl Index<GradientRef> for GradientTape {
    type Output = ArrayD<f32>;
    fn index(&self, gradient_ref: GradientRef) -> &Self::Output {
        &self.gradients[gradient_ref.index]
    }
}

#[derive(Debug)]
pub struct Grad {
    pub gradient_ref: GradientRef,
    tape: Option<Box<GradientTape>>,
}

impl Grad {
    pub fn new(gradient_ref: GradientRef) -> Self {
        Self {
            gradient_ref,
            tape: None,
        }
    }

    pub fn with_tape(gradient_ref: GradientRef, tape: Box<GradientTape>) -> Self {
        Self {
            gradient_ref,
            tape: Some(tape),
        }
    }

    pub fn keep_tape(&mut self, tape: Box<GradientTape>) {
        self.tape = Some(tape);
    }

    pub fn take_tape(&mut self) -> Option<Box<GradientTape>> {
        self.tape.take()
    }
}
