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
    };
}

macro_rules! unary_ops {
    ([$($const_defs:tt)*] $typename:ident [$($consts:tt)*], $num_elems:expr) => {
        impl<$($const_defs)*> $typename<$($consts)*> {
            pub fn relu(&mut self) -> Self {
                let grad = self.take_tape().map(|mut tape| {
                    self.register(&mut tape);

                    let parent_deriv =
                        tape.store_derivative(self.data.mapv(|f| if f >= 0.0 { 1.0 } else { 0.0 }));
                    let result_grad = tape.store_gradient(Self::SHAPE);

                    tape.add_operation(Operation::Unary(UnaryOp {
                        op_type: OpType::ReLU,
                        parent_grad: self.gradient_ref(),
                        parent_deriv,
                        result_grad,
                    }));

                    Grad::with_tape(result_grad, tape)
                });

                Self {
                    data: self.data.map(|&f| 0.0f32.max(f)),
                    grad,
                }
            }

            pub fn square(&mut self) -> Self {
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

                Self {
                    data: self.data.map(|f| f.powi(2)),
                    grad,
                }
            }

            pub fn mean(&mut self) -> Tensor0D {
                let grad = self.take_tape().map(|mut tape| {
                    self.register(&mut tape);

                    let parent_deriv = tape.store_derivative(Array::from_elem(Self::SHAPE, 1.0 / $num_elems as f32));
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
    };
}

binary_ops!([] Tensor0D []);
binary_ops!([const N: usize] Tensor1D [N]);
binary_ops!([const M: usize, const N: usize] Tensor2D [M, N]);

unary_ops!([] Tensor0D [], 1.0);
unary_ops!([const N: usize] Tensor1D [N], N);
unary_ops!([const M: usize, const N: usize] Tensor2D [M, N], M * N);

impl<const M: usize, const N: usize> Mul<&mut Tensor1D<N>> for &mut Tensor2D<M, N> {
    type Output = Tensor1D<M>;
    fn mul(self, rhs: &mut Tensor1D<N>) -> Self::Output {
        let grad = self.take_tape().or(rhs.take_tape()).map(|mut tape| {
            self.register(&mut tape);
            rhs.register(&mut tape);

            let lhs_deriv = tape.store_derivative(rhs.data.clone());
            let rhs_deriv = tape.store_derivative(self.data.clone());
            let result_grad = tape.store_gradient(Self::Output::SHAPE);

            tape.add_operation(Operation::Binary(BinaryOp {
                op_type: OpType::MatMul { m: M, n: N, o: 1 },
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
                op_type: OpType::BroadcastAdd,
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
