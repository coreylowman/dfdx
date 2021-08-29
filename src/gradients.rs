use std::ops::Index;

use ndarray::prelude::*;

#[derive(Debug)]
pub enum Op {
    Add,
    Sub,
    Matmul { m: usize, n: usize },
    Square,
    Mean,
}

#[derive(Debug)]
struct UnaryOp {
    op: Op,
    parent: usize,
    deriv: usize,
    result: usize,
}

#[derive(Debug)]
struct BinaryOp {
    op: Op,
    parents: [usize; 2],
    derivs: [usize; 2],
    result: usize,
}

#[derive(Debug)]
enum Operation {
    Advance,
    Unary(UnaryOp),
    Binary(BinaryOp),
}

#[derive(Debug)]
pub struct GradientTape {
    num_tensors: usize,
    operations: Vec<Operation>,
    derivatives: Vec<ArrayD<f32>>,
    gradients: Vec<ArrayD<f32>>,
}

impl GradientTape {
    pub fn new() -> Self {
        Self {
            num_tensors: 0,
            operations: Vec::new(),
            derivatives: Vec::new(),
            gradients: Vec::new(),
        }
    }

    pub fn scale(&mut self, scalar: f32) {
        for g in self.gradients.iter_mut() {
            *g *= scalar;
        }
    }

    pub fn advance(&mut self, result_shape: &[usize]) -> usize {
        let index = self.num_tensors;
        self.num_tensors += 1;
        self.gradients.push(ArrayD::zeros(result_shape));
        self.operations.push(Operation::Advance);
        index
    }

    pub fn unary_op(
        &mut self,
        op: Op,
        parent: usize,
        deriv: ArrayD<f32>,
        result_shape: &[usize],
    ) -> usize {
        let result_index = self.advance(result_shape);

        let deriv_index = self.derivatives.len();
        self.derivatives.push(deriv);
        self.operations.push(Operation::Unary(UnaryOp {
            op,
            parent,
            result: result_index,
            deriv: deriv_index,
        }));

        result_index
    }

    pub fn binary_op(
        &mut self,
        op: Op,
        lhs_parent: usize,
        rhs_parent: usize,
        lhs_deriv: ArrayD<f32>,
        rhs_deriv: ArrayD<f32>,
        result_shape: &[usize],
    ) -> usize {
        let result_index = self.advance(result_shape);

        let deriv_index = self.derivatives.len();
        self.derivatives.push(lhs_deriv);
        self.derivatives.push(rhs_deriv);
        self.operations.push(Operation::Binary(BinaryOp {
            op,
            parents: [lhs_parent, rhs_parent],
            result: result_index,
            derivs: [deriv_index, deriv_index + 1],
        }));

        result_index
    }

    pub fn backward(&mut self, tag: Option<usize>) {
        let index: usize = tag.unwrap();
        self.gradients[index].fill(1.0);
        for operation in self.operations.iter().rev() {
            match operation {
                Operation::Advance => {}
                Operation::Unary(op) => {
                    let d_grad = &self.derivatives[op.deriv] * &self.gradients[op.result];
                    self.gradients[op.parent] += &d_grad;
                }
                Operation::Binary(op) => match op.op {
                    Op::Matmul { m, n } => {
                        let d_grad = (&self.gradients[op.result] * &self.derivatives[op.derivs[0]])
                            .reversed_axes();
                        self.gradients[op.parents[0]] += &d_grad;

                        let wt = (&self.derivatives[op.derivs[1]])
                            .clone()
                            .into_shape((n, m))
                            .expect("");
                        let x = (&self.gradients[op.result])
                            .clone()
                            .into_shape((m, 1))
                            .expect("");
                        let d_grad = wt.dot(&x).into_shape((n,)).expect("");
                        self.gradients[op.parents[1]] += &d_grad;
                    }
                    _ => {
                        for i in 0..2 {
                            let d_grad =
                                &self.derivatives[op.derivs[i]] * &self.gradients[op.result];
                            self.gradients[op.parents[i]] += &d_grad;
                        }
                    }
                },
            }
        }
    }
}

#[derive(Debug)]
pub struct GradientRef {
    tag: Option<usize>,
    tape: Option<Box<GradientTape>>,
}

impl GradientRef {
    pub fn set_tag(&mut self, tag: Option<usize>) {
        self.tag = tag;
    }

    pub fn has_tag(&self) -> bool {
        self.tag.is_some()
    }

    pub fn tag(&self) -> usize {
        self.tag.unwrap()
    }

    pub fn keep_tape(&mut self, tape: Option<Box<GradientTape>>) {
        self.tape = tape;
    }

    pub fn take_tape(&mut self) -> Option<Box<GradientTape>> {
        self.tape.take()
    }

    pub fn backward(&mut self) -> Box<GradientTape> {
        let mut tape = self.tape.take().unwrap();
        tape.backward(self.tag);
        tape
    }
}

impl Default for GradientRef {
    fn default() -> Self {
        Self {
            tag: None,
            tape: None,
        }
    }
}

impl Index<usize> for GradientTape {
    type Output = ArrayD<f32>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.gradients[index]
    }
}
