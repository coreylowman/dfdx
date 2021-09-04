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

#[derive(Debug)]
pub enum OpType {
    Add,
    Sub,
    MatVec { m: usize, n: usize },
    MatMul { m: usize, n: usize, o: usize },
    Square,
    Mean,
}

#[derive(Debug)]
pub struct UnaryOp {
    pub op_type: OpType,
    pub parent_grad: GradientRef,
    pub parent_deriv: DerivativeRef,
    pub result_grad: GradientRef,
}

#[derive(Debug)]
pub struct BinaryOp {
    pub op_type: OpType,
    pub parent_grads: [GradientRef; 2],
    pub parent_derivs: [DerivativeRef; 2],
    pub result_grad: GradientRef,
}

#[derive(Debug)]
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

    pub fn store_gradient(&mut self, shape: &[usize]) -> GradientRef {
        let index = self.gradients.len();
        self.gradients.push(ArrayD::zeros(shape));
        GradientRef { index }
    }

    pub fn add_operation(&mut self, operation: Operation) {
        self.operations.push(operation);
    }

    pub fn backward(&mut self, gradient_ref: GradientRef) {
        self.gradients[gradient_ref.index].fill(1.0);
        for operation in self.operations.iter().rev() {
            match operation {
                Operation::Unary(op) => {
                    let d_grad = &self.derivatives[op.parent_deriv.index]
                        * &self.gradients[op.result_grad.index];
                    self.gradients[op.parent_grad.index] += &d_grad;
                }
                Operation::Binary(op) => match op.op_type {
                    OpType::MatVec { m, n } => {
                        let d_grad = (&self.gradients[op.result_grad.index]
                            * &self.derivatives[op.parent_derivs[0].index])
                            .reversed_axes();
                        self.gradients[op.parent_grads[0].index] += &d_grad;

                        let wt = (&self.derivatives[op.parent_derivs[1].index])
                            .clone()
                            .into_shape((n, m))
                            .expect("");
                        let x = (&self.gradients[op.result_grad.index])
                            .clone()
                            .into_shape((m, 1))
                            .expect("");
                        let d_grad = wt.dot(&x).into_shape((n,)).expect("");
                        self.gradients[op.parent_grads[1].index] += &d_grad;
                    }
                    _ => {
                        for i in 0..2 {
                            let d_grad = &self.derivatives[op.parent_derivs[i].index]
                                * &self.gradients[op.result_grad.index];
                            self.gradients[op.parent_grads[i].index] += &d_grad;
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
