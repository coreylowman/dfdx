use super::structs::Tensor0D;
use super::traits::{Mean, Tensor};
use crate::diff_fns::*;
use crate::gradients::{Gradient, Operation, UnaryOp};
use ndarray::prelude::*;

impl<T> Mean for T
where
    T: Tensor,
{
    fn mean(&mut self) -> Tensor0D {
        let grad = self.take_tape().map(|mut tape| {
            let parent_deriv =
                tape.store_derivative(self.data().mapv(|_f| 1.0 / Self::NUM_ELEMENTS as f32));
            let result_grad = tape.register_gradient(());

            tape.add_operation(Operation::Unary(UnaryOp {
                parent_grad: self.gradient_ref(),
                parent_deriv,
                result_grad,
            }));

            Gradient::on_tape(result_grad, tape)
        });

        Tensor0D {
            data: arr0(self.data().mean().unwrap()),
            grad,
        }
    }
}

impl<T> ApplyDifferentiableFunction for T
where
    T: Tensor,
{
    fn apply<F: DifferentiableFunction>(&mut self) -> Self {
        let mut output = Self::default();
        *output.mut_data() = self.data().mapv(F::f);
        *output.mut_grad() = self.take_tape().map(|mut tape| {
            let parent_deriv = tape.store_derivative(self.data().mapv(F::df));
            let result_grad = tape.register_gradient(Self::SHAPE);

            tape.add_operation(Operation::Unary(UnaryOp {
                parent_grad: self.gradient_ref(),
                parent_deriv,
                result_grad,
            }));

            Gradient::on_tape(result_grad, tape)
        });
        output
    }
}
