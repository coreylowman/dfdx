use crate::gradients::{Operation, UnaryOp};
use crate::prelude::*;
use ndarray::prelude::*;

pub(super) fn add_unary_op<Inp, Out, D>(
    tape: &mut Box<GradientTape>,
    operands: (&Inp, &Out),
    deriv: Array<f32, D>,
) where
    Inp: HasUniqueId + IsShapedArray,
    Out: HasUniqueId + IsShapedArray,
    D: Dimension,
{
    let parent_grad = tape.gradient_ref_for(operands.0.id(), operands.0.shape());
    let parent_deriv = tape.store_derivative(deriv);
    let result_grad = tape.gradient_ref_for(operands.1.id(), operands.1.shape());
    tape.add_operation(Operation::Unary(UnaryOp {
        parent_grad,
        parent_deriv,
        result_grad,
    }));
}

pub trait HasMeanMethod: Tensor {
    fn mean(self) -> Tensor0D<Self::TapeHolder>;
}

impl<T: Tensor> HasMeanMethod for T {
    fn mean(self) -> Tensor0D<Self::TapeHolder> {
        let result = Tensor0D::<NoTape>::new(arr0(self.data().mean().unwrap()));
        let (t, mut tape_holder) = self.split_tape_holder();
        tape_holder.update_with(|tape| {
            add_unary_op(
                tape,
                (&t, &result),
                t.data().mapv(|_| 1.0 / T::NUM_ELEMENTS as f32),
            )
        });
        result.with_tape_holder(tape_holder)
    }
}

pub fn apply<T: Tensor, F: DifferentiableFunction>(t: T) -> T {
    let result = T::NoTape::new(t.data().mapv(F::f));
    let (t, mut tape_holder) = t.split_tape_holder();
    tape_holder.update_with(|tape| add_unary_op(tape, (&t, &result), t.data().mapv(F::df)));
    result.with_tape_holder(tape_holder)
}

macro_rules! apply_impl {
    ($trait_name:tt, $method_name:tt, $activation_struct:ty) => {
        pub trait $trait_name {
            fn $method_name(self) -> Self;
        }

        impl<T: Tensor> $trait_name for T {
            fn $method_name(self) -> Self {
                apply::<Self, $activation_struct>(self)
            }
        }
    };
}

apply_impl!(HasReLUMethod, relu, ReLU);
apply_impl!(HasSinMethod, sin, Sin);
apply_impl!(HasCosMethod, cos, Cos);
apply_impl!(HasLnMethod, ln, Ln);
apply_impl!(HasExpMethod, exp, Exp);
apply_impl!(HasSigmoidMethod, sigmoid, Sigmoid);
apply_impl!(HasTanhMethod, tanh, Tanh);
apply_impl!(HasSquareMethod, square, Square);
apply_impl!(HasAbsMethod, abs, Abs);
