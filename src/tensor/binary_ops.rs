use super::base::*;
use crate::{
    gradients::*,
    traits::{Params, ShapedArray, Tensor},
};
use ndarray::prelude::*;
use std::ops::{Add, Mul, Sub};

macro_rules! binary_ops {
    ([$($const_defs:tt)*] $typename:ident [$($consts:tt)*]) => {
        impl<$($const_defs)*> Add for &mut $typename<$($consts)*> {
            type Output = $typename<$($consts)*>;
            fn add(self, rhs: Self) -> Self::Output {
                let grad = self.take_tape().or(rhs.take_tape()).map(|mut tape| {
                    self.register(&mut tape);
                    rhs.register(&mut tape);

                    let lhs_deriv = tape.store_derivative(Array::from_elem(Self::Output::SHAPE, 1.0));
                    let rhs_deriv = tape.store_derivative(Array::from_elem(Self::Output::SHAPE, 1.0));
                    let result_grad = tape.store_gradient(Self::Output::SHAPE);

                    tape.add_operation(Operation::Binary(BinaryOp {
                        op_type: OpType::Normal,
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

        impl<$($const_defs)*> Sub for &mut $typename<$($consts)*> {
            type Output = $typename<$($consts)*>;
            fn sub(self, rhs: Self) -> Self::Output {
                let grad = self.take_tape().or(rhs.take_tape()).map(|mut tape| {
                    self.register(&mut tape);
                    rhs.register(&mut tape);

                    let lhs_deriv = tape.store_derivative(Array::from_elem(Self::Output::SHAPE, 1.0));
                    let rhs_deriv = tape.store_derivative(Array::from_elem(Self::Output::SHAPE, -1.0));
                    let result_grad = tape.store_gradient(Self::Output::SHAPE);

                    tape.add_operation(Operation::Binary(BinaryOp {
                        op_type: OpType::Normal,
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
    };
}

binary_ops!([] Tensor0D []);
binary_ops!([const N: usize] Tensor1D [N]);
binary_ops!([const M: usize, const N: usize] Tensor2D [M, N]);

impl<const M: usize, const N: usize, const O: usize> Mul<&mut Tensor2D<N, O>>
    for &mut Tensor2D<M, N>
{
    type Output = Tensor2D<M, O>;
    fn mul(self, rhs: &mut Tensor2D<N, O>) -> Self::Output {
        let grad = self.take_tape().or(rhs.take_tape()).map(|mut tape| {
            self.register(&mut tape);
            rhs.register(&mut tape);

            let lhs_deriv = tape.store_derivative(rhs.data.clone());
            let rhs_deriv = tape.store_derivative(self.data.clone());
            let result_grad = tape.store_gradient(Self::Output::SHAPE);

            tape.add_operation(Operation::Binary(BinaryOp {
                op_type: OpType::MatMul { m: M, n: N, o: O },
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

impl<const M: usize, const N: usize> Add<&mut Tensor1D<N>> for &mut Tensor2D<M, N> {
    type Output = Tensor2D<M, N>;
    fn add(self, rhs: &mut Tensor1D<N>) -> Self::Output {
        let grad = self.take_tape().or(rhs.take_tape()).map(|mut tape| {
            self.register(&mut tape);
            rhs.register(&mut tape);

            let lhs_deriv = tape.store_derivative(Array::from_elem((M, N), 1.0));
            let rhs_deriv = tape.store_derivative(Array::from_elem((N,), 1.0 / M as f32));
            let result_grad = tape.store_gradient(Self::Output::SHAPE);

            tape.add_operation(Operation::Binary(BinaryOp {
                op_type: OpType::Broadcast,
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
