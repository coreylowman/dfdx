use super::structs::{GradientData, Tensor0D};
use super::traits::{Mean, Tensor};
use crate::diff_fns::*;
use crate::gradients::{Operation, UnaryOp};
use ndarray::prelude::*;

impl<T> Mean for T
where
    T: Tensor,
{
    fn mean(&self) -> Tensor0D {
        let mut tape = self.take_tape();
        let grad_ref = tape.as_mut().map(|mut tape| {
            let parent_grad = self.grad_ref(&mut tape);

            let parent_deriv =
                tape.store_derivative(self.data().mapv(|_| 1.0 / Self::NUM_ELEMENTS as f32));

            let result_grad = tape.allocate_gradient(());
            tape.add_operation(Operation::Unary(UnaryOp {
                parent_grad,
                parent_deriv,
                result_grad,
            }));

            result_grad
        });

        Tensor0D::new(
            arr0(self.data().mean().unwrap()),
            GradientData { grad_ref, tape },
        )
    }
}

impl<T> ApplyDifferentiableFunction for T
where
    T: Tensor,
{
    fn apply<F: DifferentiableFunction>(&self) -> Self {
        let mut tape = self.take_tape();
        let grad_ref = tape.as_mut().map(|mut tape| {
            let parent_grad = self.grad_ref(&mut tape);

            let parent_deriv = tape.store_derivative(self.data().mapv(F::df));
            let result_grad = tape.allocate_gradient(Self::SHAPE);

            tape.add_operation(Operation::Unary(UnaryOp {
                parent_grad,
                parent_deriv,
                result_grad,
            }));

            result_grad
        });

        Self::new(self.data().mapv(F::f), GradientData { grad_ref, tape })
    }
}
