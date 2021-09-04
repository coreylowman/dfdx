use crate::{
    gradients::*,
    traits::{Params, Tensor},
};
use ndarray::prelude::*;
use std::ops::{Add, Mul, Sub};

#[derive(Debug)]
pub struct Tensor0D {
    data: Array0<f32>,
    grad: Option<Grad>,
}

impl Default for Tensor0D {
    fn default() -> Self {
        Self {
            data: Array0::zeros(()),
            grad: None,
        }
    }
}

impl Tensor for Tensor0D {
    const SHAPE: &'static [usize] = &[];
    type Dimension = Ix0;

    fn grad(&self) -> &Option<Grad> {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut Option<Grad> {
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
    grad: Option<Grad>,
}

impl<const N: usize> Default for Tensor1D<N> {
    fn default() -> Self {
        Self {
            data: Array1::zeros((N,)),
            grad: None,
        }
    }
}

impl<const N: usize> Tensor for Tensor1D<N> {
    type Dimension = Ix1;
    const SHAPE: &'static [usize] = &[N];

    fn grad(&self) -> &Option<Grad> {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut Option<Grad> {
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
        let grad = self.take_tape().or(rhs.take_tape()).map(|mut tape| {
            self.register(&mut tape);
            rhs.register(&mut tape);

            let lhs_deriv = tape.store_derivative(Array1::from_elem((N,), 1.0));
            let rhs_deriv = tape.store_derivative(Array1::from_elem((N,), 1.0));
            let result_grad = tape.store_gradient(&[N]);

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

            let lhs_deriv = tape.store_derivative(Array1::from_elem((N,), 1.0));
            let rhs_deriv = tape.store_derivative(Array1::from_elem((N,), -1.0));
            let result_grad = tape.store_gradient(&[N]);

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
            let result_grad = tape.store_gradient(&[N]);

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

            let parent_deriv = tape.store_derivative(Array1::from_elem((N,), 1.0 / N as f32));
            let result_grad = tape.store_gradient(&[]);

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

#[derive(Debug)]
pub struct Tensor2D<const M: usize, const N: usize> {
    data: Array2<f32>,
    grad: Option<Grad>,
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

    fn grad(&self) -> &Option<Grad> {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut Option<Grad> {
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
        let grad = self.take_tape().or(rhs.take_tape()).map(|mut tape| {
            self.register(&mut tape);
            rhs.register(&mut tape);

            let lhs_deriv = tape.store_derivative(rhs.data.clone().into_shape((N, 1)).expect(""));
            let rhs_deriv = tape.store_derivative(self.data.clone().reversed_axes());
            let result_grad = tape.store_gradient(&[M]);

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
