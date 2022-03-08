use super::structs::Tensor0D;
use super::traits::{Mean, Tensor};
use crate::diff_fns::*;
use crate::gradients::{Operation, UnaryOp};
use ndarray::prelude::*;

impl<T> Mean for T
where
    T: Tensor,
{
    fn mean(&mut self) -> Tensor0D {
        let mut opt_tape = self.mut_tape().take();
        let opt_grad = opt_tape.as_mut().map(|mut tape| {
            self.put_on(&mut tape);

            let parent_deriv =
                tape.store_derivative(self.data().mapv(|_| 1.0 / Self::NUM_ELEMENTS as f32));

            let result_grad = tape.allocate_gradient(());
            tape.add_operation(Operation::Unary(UnaryOp {
                parent_grad: self.grad_ref().unwrap(),
                parent_deriv,
                result_grad,
            }));

            result_grad
        });

        Tensor0D::new(arr0(self.data().mean().unwrap()), opt_grad, opt_tape)
    }
}

impl<T> ApplyDifferentiableFunction for T
where
    T: Tensor,
{
    fn apply<F: DifferentiableFunction>(&mut self) -> Self {
        let mut opt_tape = self.mut_tape().take();
        let opt_grad = opt_tape.as_mut().map(|mut tape| {
            self.put_on(&mut tape);

            let parent_deriv = tape.store_derivative(self.data().mapv(F::df));
            let result_grad = tape.allocate_gradient(Self::SHAPE);

            tape.add_operation(Operation::Unary(UnaryOp {
                parent_grad: self.grad_ref().unwrap(),
                parent_deriv,
                result_grad,
            }));

            result_grad
        });

        Self::new(self.data().mapv(F::f), opt_grad, opt_tape)
    }
}
