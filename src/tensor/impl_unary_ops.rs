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
        let grad = self.take_tape().map(|mut tape| {
            self.put_on(&mut tape);

            let parent_deriv =
                tape.store_derivative(self.data().mapv(|_f| 1.0 / Self::NUM_ELEMENTS as f32));
            let mut gradient = tape.allocate_gradient(());

            tape.add_operation(Operation::Unary(UnaryOp {
                parent_grad: self.gradient_ref(),
                parent_deriv,
                result_grad: gradient.gradient_ref,
            }));

            gradient.tape = Some(tape);
            gradient
        });

        Tensor0D::new(arr0(self.data().mean().unwrap()), grad)
    }
}

impl<T> ApplyDifferentiableFunction for T
where
    T: Tensor,
{
    fn apply<F: DifferentiableFunction>(&mut self) -> Self {
        let grad = self.take_tape().map(|mut tape| {
            self.put_on(&mut tape);

            let parent_deriv = tape.store_derivative(self.data().mapv(F::df));
            let mut gradient = tape.allocate_gradient(Self::SHAPE);

            tape.add_operation(Operation::Unary(UnaryOp {
                parent_grad: self.gradient_ref(),
                parent_deriv,
                result_grad: gradient.gradient_ref,
            }));

            gradient.tape = Some(tape);
            gradient
        });
        Self::new(self.data().mapv(F::f), grad)
    }
}
