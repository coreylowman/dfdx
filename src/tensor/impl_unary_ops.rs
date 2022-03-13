use crate::gradients::{Operation, UnaryOp};
use crate::prelude::*;
use ndarray::prelude::*;

fn unary_op<T: Tensor, O: Tensor, D: Dimension>(
    mut tape: Box<GradientTape>,
    parent: &T,
    result: &O,
    deriv: Array<f32, D>,
) -> Box<GradientTape> {
    let parent_grad = tape.gradient_ref_for(parent.id(), parent.shape());
    let parent_deriv = tape.store_derivative(deriv);
    let result_grad = tape.gradient_ref_for(result.id(), result.shape());
    tape.add_operation(Operation::Unary(UnaryOp {
        parent_grad,
        parent_deriv,
        result_grad,
    }));
    tape
}

impl<T: Tensor> Mean for T {
    fn mean(&self) -> Tensor0D {
        let result = Tensor0D::new(arr0(self.data().mean().unwrap()));
        result.put_tape(self.take_tape().map(|tape| {
            unary_op(
                tape,
                self,
                &result,
                self.data().mapv(|_| 1.0 / Self::NUM_ELEMENTS as f32),
            )
        }));
        result
    }
}

impl<T: Tensor> ApplyDifferentiableFunction for T {
    fn apply<F: DifferentiableFunction>(&self) -> Self {
        let result = Self::new(self.data().mapv(F::f));
        result.put_tape(
            self.take_tape()
                .map(|tape| unary_op(tape, self, &result, self.data().mapv(F::df))),
        );
        result
    }
}
