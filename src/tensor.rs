use crate::{
    gradients::*,
    traits::{Params, Tensor},
};
use ndarray::prelude::*;
use std::ops::{Add, Mul, Sub};

// #[derive(Default, Debug)]
// struct TensorBase<D: Dimension> {
//     data: Array<f32, D>,
//     grad: Grad,
// }

#[derive(Default, Debug)]
pub struct Tensor0D {
    data: Array0<f32>,
    grad: Grad,
}

impl Tensor for Tensor0D {
    const SHAPE: &'static [usize] = &[];
    type Dimension = Ix0;

    fn grad(&self) -> &Grad {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut Grad {
        &mut self.grad
    }

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }

    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

#[derive(Debug)]
pub struct Tensor1D<const N: usize> {
    data: Array1<f32>,
    grad: Grad,
}

impl<const N: usize> Default for Tensor1D<N> {
    fn default() -> Self {
        Self {
            data: Array1::zeros((N,)),
            grad: Default::default(),
        }
    }
}

impl<const N: usize> Tensor for Tensor1D<N> {
    type Dimension = Ix1;
    const SHAPE: &'static [usize] = &[N];

    fn grad(&self) -> &Grad {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut Grad {
        &mut self.grad
    }

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }

    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

impl<const N: usize> Add for &mut Tensor1D<N> {
    type Output = Tensor1D<N>;
    fn add(self, rhs: &mut Tensor1D<N>) -> Self::Output {
        let mut result = Tensor1D {
            data: &self.data + &rhs.data,
            grad: Default::default(),
        };

        let mut opt_tape = self.take_tape().or(rhs.take_tape());
        opt_tape.as_mut().map(|tape| {
            self.register(tape);
            rhs.register(tape);

            let lhs_deriv = tape.store_derivative(Array1::from_elem((N,), 1.0));
            let rhs_deriv = tape.store_derivative(Array1::from_elem((N,), 1.0));
            let result_grad = tape.store_gradient(&[N]);

            tape.add_operation(Operation::Binary(BinaryOp {
                op_type: OpType::Add,
                parent_grads: [self.gradient_ref(), rhs.gradient_ref()],
                parent_derivs: [lhs_deriv, rhs_deriv],
                result_grad,
            }));

            result.mut_grad().set_gradient_ref(result_grad);
        });
        result.keep_tape(opt_tape);

        result
    }
}

impl<const N: usize> Sub for &mut Tensor1D<N> {
    type Output = Tensor1D<N>;
    fn sub(self, rhs: &mut Tensor1D<N>) -> Self::Output {
        let mut result = Tensor1D {
            data: &self.data - &rhs.data,
            grad: Default::default(),
        };

        let mut opt_tape = self.take_tape().or(rhs.take_tape());
        opt_tape.as_mut().map(|tape| {
            self.register(tape);
            rhs.register(tape);

            let lhs_deriv = tape.store_derivative(Array1::from_elem((N,), 1.0));
            let rhs_deriv = tape.store_derivative(Array1::from_elem((N,), -1.0));
            let result_grad = tape.store_gradient(&[N]);

            tape.add_operation(Operation::Binary(BinaryOp {
                op_type: OpType::Sub,
                parent_grads: [self.gradient_ref(), rhs.gradient_ref()],
                parent_derivs: [lhs_deriv, rhs_deriv],
                result_grad,
            }));

            result.mut_grad().set_gradient_ref(result_grad);
        });
        result.keep_tape(opt_tape);

        result
    }
}

impl<const N: usize> Tensor1D<N> {
    pub fn square(&mut self) -> Tensor1D<N> {
        let mut result = Tensor1D {
            data: self.data.map(|f| f.powi(2)),
            grad: Default::default(),
        };

        let mut opt_tape = self.take_tape();
        opt_tape.as_mut().map(|tape| {
            self.register(tape);

            let parent_deriv = tape.store_derivative(2.0 * &self.data);
            let result_grad = tape.store_gradient(&[N]);

            tape.add_operation(Operation::Unary(UnaryOp {
                op_type: OpType::Square,
                parent_grad: self.gradient_ref(),
                parent_deriv,
                result_grad,
            }));

            result.mut_grad().set_gradient_ref(result_grad);
        });
        result.keep_tape(opt_tape);

        result
    }

    pub fn mean(&mut self) -> Tensor0D {
        let mut result = Tensor0D {
            data: arr0(self.data.mean().unwrap()),
            grad: Default::default(),
        };

        let mut opt_tape = self.take_tape();
        opt_tape.as_mut().map(|tape| {
            self.register(tape);

            let parent_deriv = tape.store_derivative(Array1::from_elem((N,), 1.0 / N as f32));
            let result_grad = tape.store_gradient(&[]);

            tape.add_operation(Operation::Unary(UnaryOp {
                op_type: OpType::Mean,
                parent_grad: self.gradient_ref(),
                parent_deriv,
                result_grad,
            }));

            result.mut_grad().set_gradient_ref(result_grad);
        });
        result.keep_tape(opt_tape);

        result
    }
}

#[derive(Debug)]
pub struct Tensor2D<const M: usize, const N: usize> {
    data: Array2<f32>,
    grad: Grad,
}

impl<const M: usize, const N: usize> Default for Tensor2D<M, N> {
    fn default() -> Self {
        Self {
            data: Array2::zeros((M, N)),
            grad: Default::default(),
        }
    }
}

impl<const M: usize, const N: usize> Tensor for Tensor2D<M, N> {
    type Dimension = Ix2;
    const SHAPE: &'static [usize] = &[M, N];

    fn grad(&self) -> &Grad {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut Grad {
        &mut self.grad
    }

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }

    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

impl<const M: usize, const N: usize> Mul<&mut Tensor1D<N>> for &mut Tensor2D<M, N> {
    type Output = Tensor1D<M>;
    fn mul(self, rhs: &mut Tensor1D<N>) -> Self::Output {
        let mut result = Tensor1D {
            data: self.data.dot(&rhs.data),
            grad: Default::default(),
        };

        let mut opt_tape = self.take_tape().or(rhs.take_tape());
        opt_tape.as_mut().map(|tape| {
            self.register(tape);
            rhs.register(tape);

            let lhs_deriv = tape.store_derivative(rhs.data.clone().into_shape((N, 1)).expect(""));
            let rhs_deriv = tape.store_derivative(self.data.clone().reversed_axes());
            let result_grad = tape.store_gradient(&[M]);

            tape.add_operation(Operation::Binary(BinaryOp {
                op_type: OpType::MatVec { m: M, n: N },
                parent_grads: [self.gradient_ref(), rhs.gradient_ref()],
                parent_derivs: [lhs_deriv, rhs_deriv],
                result_grad,
            }));

            result.mut_grad().set_gradient_ref(result_grad);
        });
        result.keep_tape(opt_tape);

        result
    }
}
