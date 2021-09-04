use super::base::*;
use crate::{
    gradients::*,
    traits::{Params, Tensor},
};
use ndarray::prelude::*;
use std::ops::{Add, Mul, Sub};

impl<const N: usize> Add for &mut Tensor1D<N> {
    type Output = Tensor1D<N>;
    fn add(self, rhs: &mut Tensor1D<N>) -> Self::Output {
        let grad = self.take_tape().or(rhs.take_tape()).map(|mut tape| {
            self.register(&mut tape);
            rhs.register(&mut tape);

            let lhs_deriv = tape.store_derivative(Array1::from_elem(Self::Output::SHAPE, 1.0));
            let rhs_deriv = tape.store_derivative(Array1::from_elem(Self::Output::SHAPE, 1.0));
            let result_grad = tape.store_gradient(Self::Output::SHAPE);

            tape.add_operation(Operation::Binary(BinaryOp {
                op_type: OpType::Add,
                parent_grads: [self.gradient_ref(), rhs.gradient_ref()],
                parent_derivs: [lhs_deriv, rhs_deriv],
                result_grad,
            }));

            Grad::with_tape(result_grad, tape)
        });

        Self::Output {
            data: &self.data + &rhs.data,
            grad,
        }
    }
}

impl<const N: usize> Sub for &mut Tensor1D<N> {
    type Output = Tensor1D<N>;
    fn sub(self, rhs: &mut Tensor1D<N>) -> Self::Output {
        let grad = self.take_tape().or(rhs.take_tape()).map(|mut tape| {
            self.register(&mut tape);
            rhs.register(&mut tape);

            let lhs_deriv = tape.store_derivative(Array1::from_elem(Self::Output::SHAPE, 1.0));
            let rhs_deriv = tape.store_derivative(Array1::from_elem(Self::Output::SHAPE, -1.0));
            let result_grad = tape.store_gradient(Self::Output::SHAPE);

            tape.add_operation(Operation::Binary(BinaryOp {
                op_type: OpType::Sub,
                parent_grads: [self.gradient_ref(), rhs.gradient_ref()],
                parent_derivs: [lhs_deriv, rhs_deriv],
                result_grad,
            }));

            Grad::with_tape(result_grad, tape)
        });

        Self::Output {
            data: &self.data - &rhs.data,
            grad,
        }
    }
}

impl<const N: usize> Tensor1D<N> {
    pub fn square(&mut self) -> Tensor1D<N> {
        let grad = self.take_tape().map(|mut tape| {
            self.register(&mut tape);

            let parent_deriv = tape.store_derivative(2.0 * &self.data);
            let result_grad = tape.store_gradient(Self::SHAPE);

            tape.add_operation(Operation::Unary(UnaryOp {
                op_type: OpType::Square,
                parent_grad: self.gradient_ref(),
                parent_deriv,
                result_grad,
            }));

            Grad::with_tape(result_grad, tape)
        });

        Tensor1D {
            data: self.data.map(|f| f.powi(2)),
            grad,
        }
    }

    pub fn mean(&mut self) -> Tensor0D {
        let grad = self.take_tape().map(|mut tape| {
            self.register(&mut tape);

            let parent_deriv =
                tape.store_derivative(Array1::from_elem(Self::SHAPE, 1.0 / N as f32));
            let result_grad = tape.store_gradient(());

            tape.add_operation(Operation::Unary(UnaryOp {
                op_type: OpType::Mean,
                parent_grad: self.gradient_ref(),
                parent_deriv,
                result_grad,
            }));

            Grad::with_tape(result_grad, tape)
        });

        Tensor0D {
            data: arr0(self.data.mean().unwrap()),
            grad,
        }
    }
}

impl<const M: usize, const N: usize> Mul<&mut Tensor1D<N>> for &mut Tensor2D<M, N> {
    type Output = Tensor1D<M>;
    fn mul(self, rhs: &mut Tensor1D<N>) -> Self::Output {
        let grad = self.take_tape().or(rhs.take_tape()).map(|mut tape| {
            self.register(&mut tape);
            rhs.register(&mut tape);

            let lhs_deriv = tape.store_derivative(rhs.data.clone().into_shape((N, 1)).expect(""));
            let rhs_deriv = tape.store_derivative(self.data.clone().reversed_axes());
            let result_grad = tape.store_gradient(Self::Output::SHAPE);

            tape.add_operation(Operation::Binary(BinaryOp {
                op_type: OpType::MatVec { m: M, n: N },
                parent_grads: [self.gradient_ref(), rhs.gradient_ref()],
                parent_derivs: [lhs_deriv, rhs_deriv],
                result_grad,
            }));

            Grad::with_tape(result_grad, tape)
        });

        Self::Output {
            data: self.data.dot(&rhs.data),
            grad,
        }
    }
}
