use crate::gradients::{Operation, UnaryOp};
use crate::prelude::*;
use ndarray::prelude::*;

impl<T: Tensor> Mean for T {
    fn mean(&self) -> Tensor0D {
        let result = Tensor0D::new(arr0(self.data().mean().unwrap()));
        let taken_tape = self.take_tape();
        let modified_tape = taken_tape.map(|mut tape| {
            let parent_grad = tape.gradient_ref_for(self.id(), self.shape());
            let parent_deriv =
                tape.store_derivative(self.data().mapv(|_| 1.0 / Self::NUM_ELEMENTS as f32));
            let result_grad = tape.gradient_ref_for(result.id(), result.shape());

            tape.add_operation(Operation::Unary(UnaryOp {
                parent_grad,
                parent_deriv,
                result_grad,
            }));

            tape
        });
        result.put_tape(modified_tape);
        result
    }
}

impl<T: Tensor> ApplyDifferentiableFunction for T {
    fn apply<F: DifferentiableFunction>(&self) -> Self {
        let result = Self::new(self.data().mapv(F::f));
        let taken_tape = self.take_tape();
        let modified_tape = taken_tape.map(|mut tape| {
            let parent_grad = tape.gradient_ref_for(self.id(), self.shape());
            let parent_deriv = tape.store_derivative(self.data().mapv(F::df));
            let result_grad = tape.gradient_ref_for(result.id(), result.shape());

            tape.add_operation(Operation::Unary(UnaryOp {
                parent_grad,
                parent_deriv,
                result_grad,
            }));

            tape
        });
        result.put_tape(modified_tape);
        result
    }
}
