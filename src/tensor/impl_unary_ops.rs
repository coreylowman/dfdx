use crate::gradients::{Operation, UnaryOp};
use crate::prelude::*;
use ndarray::prelude::*;

fn unary_op<T: HasUniqueId + IsShapedArray, O: HasUniqueId + IsShapedArray, D: Dimension>(
    mut tape: Box<GradientTape>,
    operands: (&T, &O),
    deriv: Array<f32, D>,
) -> Box<GradientTape> {
    let parent_grad = tape.gradient_ref_for(operands.0.id(), operands.0.shape());
    let parent_deriv = tape.store_derivative(deriv);
    let result_grad = tape.gradient_ref_for(operands.1.id(), operands.1.shape());
    tape.add_operation(Operation::Unary(UnaryOp {
        parent_grad,
        parent_deriv,
        result_grad,
    }));
    tape
}

fn mean_no_tape<T: TensorNoTape>(t: T) -> Tensor0D<NoTape> {
    Tensor0D::new_no_tape(arr0(t.data().mean().unwrap()))
}

fn mean_with_tape<T: TensorWithTape>(t: T) -> Tensor0D<WithTape> {
    let (no_tape, tape) = t.without_tape();
    let result = Tensor0D::new_no_tape(arr0(no_tape.data().mean().unwrap()));
    let tape = unary_op(
        tape,
        (&no_tape, &result),
        no_tape.data().mapv(|_| 1.0 / T::NUM_ELEMENTS as f32),
    );
    result.put_tape(tape)
}

pub(crate) fn apply_no_tape<T: TensorNoTape, F: DifferentiableFunction>(t: T) -> T {
    T::new_no_tape(t.data().mapv(F::f))
}

pub(crate) fn apply_with_tape<T: TensorWithTape, F: DifferentiableFunction>(t: T) -> T {
    let (no_tape, tape) = t.without_tape();
    let result = T::NoTape::new_no_tape(no_tape.data().mapv(F::f));
    let tape = unary_op(tape, (&no_tape, &result), no_tape.data().mapv(F::df));
    result.put_tape(tape)
}

macro_rules! unary_impl {
    ($typename:ident, [$($const_names:tt),*]) => {
        impl<$(const $const_names: usize),*> $typename<$($const_names, )* NoTape> {
            pub fn mean(self) -> Tensor0D<NoTape> {
                mean_no_tape(self)
            }

            pub fn relu(self) -> Self {
                apply_no_tape::<Self, ReLU>(self)
            }

            pub fn sin(self) -> Self {
                apply_no_tape::<Self, Sin>(self)
            }

            pub fn cos(self) -> Self {
                apply_no_tape::<Self, Cos>(self)
            }

            pub fn ln(self) -> Self {
                apply_no_tape::<Self, Ln>(self)
            }

            pub fn exp(self) -> Self {
                apply_no_tape::<Self, Exp>(self)
            }

            pub fn sigmoid(self) -> Self {
                apply_no_tape::<Self, Sigmoid>(self)
            }

            pub fn tanh(self) -> Self {
                apply_no_tape::<Self, Tanh>(self)
            }

            pub fn square(self) -> Self {
                apply_no_tape::<Self, Square>(self)
            }

            pub fn abs(self) -> Self {
                apply_no_tape::<Self, Abs>(self)
            }
        }

        impl<$(const $const_names: usize),*> $typename<$($const_names, )* WithTape> {
            pub fn mean(self) -> Tensor0D<WithTape> {
                mean_with_tape(self)
            }

            pub fn relu(self) -> Self {
                apply_with_tape::<Self, ReLU>(self)
            }

            pub fn sin(self) -> Self {
                apply_with_tape::<Self, Sin>(self)
            }

            pub fn cos(self) -> Self {
                apply_with_tape::<Self, Cos>(self)
            }

            pub fn ln(self) -> Self {
                apply_with_tape::<Self, Ln>(self)
            }

            pub fn exp(self) -> Self {
                apply_with_tape::<Self, Exp>(self)
            }

            pub fn sigmoid(self) -> Self {
                apply_with_tape::<Self, Sigmoid>(self)
            }

            pub fn tanh(self) -> Self {
                apply_with_tape::<Self, Tanh>(self)
            }

            pub fn square(self) -> Self {
                apply_with_tape::<Self, Square>(self)
            }

            pub fn abs(self) -> Self {
                apply_with_tape::<Self, Abs>(self)
            }
        }
    };
}

unary_impl!(Tensor0D, []);
unary_impl!(Tensor1D, [N]);
unary_impl!(Tensor2D, [M, N]);
unary_impl!(Tensor3D, [M, N, O]);
unary_impl!(Tensor4D, [M, N, O, P]);
