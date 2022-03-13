use super::structs::Tensor0D;
use super::traits::{HasUniqueId, Mean, Tensor};
use super::CanStoreGradientTape;
use crate::diff_fns::*;
use crate::gradients::{Operation, UnaryOp};
use ndarray::prelude::*;

impl<T> Mean for T
where
    T: Tensor,
{
    fn mean(&self) -> Tensor0D {
        let result = Tensor0D::new(arr0(self.data().mean().unwrap()));
        result.put_tape(self.take_tape().map(|mut tape| {
            let parent_grad = tape.gradient_ref_for(self.id(), Self::SHAPE);

            let parent_deriv =
                tape.store_derivative(self.data().mapv(|_| 1.0 / Self::NUM_ELEMENTS as f32));

            let result_grad = tape.gradient_ref_for(result.id(), ());
            tape.add_operation(Operation::Unary(UnaryOp {
                parent_grad,
                parent_deriv,
                result_grad,
            }));
            tape
        }));
        result
    }
}

impl<T> ApplyDifferentiableFunction for T
where
    T: Tensor,
{
    fn apply<F: DifferentiableFunction>(&self) -> Self {
        let result = Self::new(self.data().mapv(F::f));
        result.put_tape(self.take_tape().map(|mut tape| {
            let parent_grad = tape.gradient_ref_for(self.id(), Self::SHAPE);

            let parent_deriv = tape.store_derivative(self.data().mapv(F::df));
            let result_grad = tape.gradient_ref_for(result.id(), Self::SHAPE);

            tape.add_operation(Operation::Unary(UnaryOp {
                parent_grad,
                parent_deriv,
                result_grad,
            }));

            tape
        }));
        result
    }
}
