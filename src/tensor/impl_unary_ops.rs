use crate::gradients::{Operation, UnaryOp};
use crate::prelude::*;
use ndarray::prelude::*;

fn unary_op<T: HasUniqueId + IsShapedArray, O: HasUniqueId + IsShapedArray, D: Dimension>(
    tape: &mut Box<GradientTape>,
    operands: (&T, &O),
    deriv: Array<f32, D>,
) {
    let parent_grad = tape.gradient_ref_for(operands.0.id(), operands.0.shape());
    let parent_deriv = tape.store_derivative(deriv);
    let result_grad = tape.gradient_ref_for(operands.1.id(), operands.1.shape());
    tape.add_operation(Operation::Unary(UnaryOp {
        parent_grad,
        parent_deriv,
        result_grad,
    }));
}

fn mean<T: Tensor>(t: T) -> Tensor0D<T::TapeManager> {
    let result = Tensor0D::<NoTape>::new(arr0(t.data().mean().unwrap()));
    let (t, mut tape_manager) = t.split_tape_manager();
    tape_manager.update_with(|tape| {
        unary_op(
            tape,
            (&t, &result),
            t.data().mapv(|_| 1.0 / T::NUM_ELEMENTS as f32),
        )
    });
    result.with_tape_manager(tape_manager)
}

macro_rules! reduction_impl {
    ($typename:ident, [$($const_names:tt),*]) => {
impl<$(const $const_names: usize, )* Mgr: TapeManager> $typename<$($const_names, )* Mgr> {
    pub fn mean(self) -> Tensor0D<Mgr> {
        mean(self)
    }
}
    };
}

reduction_impl!(Tensor0D, []);
reduction_impl!(Tensor1D, [N]);
reduction_impl!(Tensor2D, [M, N]);
reduction_impl!(Tensor3D, [M, N, O]);
reduction_impl!(Tensor4D, [M, N, O, P]);

pub(crate) fn apply<T: Tensor, F: DifferentiableFunction>(t: T) -> T
where
    T::NoTape: TensorCreator + CanAddTape<T::TapeManager, Output = T>,
{
    let result = T::NoTape::new(t.data().mapv(F::f));
    let (t, mut tape_manager) = t.split_tape_manager();
    tape_manager.update_with(|tape| unary_op(tape, (&t, &result), t.data().mapv(F::df)));
    result.with_tape_manager(tape_manager)
}

macro_rules! apply_impl {
    ($typename:ident, $method_name:tt, $activation_struct:ty, [$($const_names:tt),*]) => {
impl<$(const $const_names: usize, )* Mgr: TapeManager> $typename<$($const_names, )* Mgr> {
    pub fn $method_name(self) -> Self {
        apply::<Self, $activation_struct>(self)
    }
}
    };
}

apply_impl!(Tensor0D, relu, ReLU, []);
apply_impl!(Tensor1D, relu, ReLU, [N]);
apply_impl!(Tensor2D, relu, ReLU, [M, N]);
apply_impl!(Tensor3D, relu, ReLU, [M, N, O]);
apply_impl!(Tensor4D, relu, ReLU, [M, N, O, P]);

apply_impl!(Tensor0D, sin, Sin, []);
apply_impl!(Tensor1D, sin, Sin, [N]);
apply_impl!(Tensor2D, sin, Sin, [M, N]);
apply_impl!(Tensor3D, sin, Sin, [M, N, O]);
apply_impl!(Tensor4D, sin, Sin, [M, N, O, P]);

apply_impl!(Tensor0D, cos, Cos, []);
apply_impl!(Tensor1D, cos, Cos, [N]);
apply_impl!(Tensor2D, cos, Cos, [M, N]);
apply_impl!(Tensor3D, cos, Cos, [M, N, O]);
apply_impl!(Tensor4D, cos, Cos, [M, N, O, P]);

apply_impl!(Tensor0D, ln, Ln, []);
apply_impl!(Tensor1D, ln, Ln, [N]);
apply_impl!(Tensor2D, ln, Ln, [M, N]);
apply_impl!(Tensor3D, ln, Ln, [M, N, O]);
apply_impl!(Tensor4D, ln, Ln, [M, N, O, P]);

apply_impl!(Tensor0D, exp, Exp, []);
apply_impl!(Tensor1D, exp, Exp, [N]);
apply_impl!(Tensor2D, exp, Exp, [M, N]);
apply_impl!(Tensor3D, exp, Exp, [M, N, O]);
apply_impl!(Tensor4D, exp, Exp, [M, N, O, P]);

apply_impl!(Tensor0D, sigmoid, Sigmoid, []);
apply_impl!(Tensor1D, sigmoid, Sigmoid, [N]);
apply_impl!(Tensor2D, sigmoid, Sigmoid, [M, N]);
apply_impl!(Tensor3D, sigmoid, Sigmoid, [M, N, O]);
apply_impl!(Tensor4D, sigmoid, Sigmoid, [M, N, O, P]);

apply_impl!(Tensor0D, tanh, Tanh, []);
apply_impl!(Tensor1D, tanh, Tanh, [N]);
apply_impl!(Tensor2D, tanh, Tanh, [M, N]);
apply_impl!(Tensor3D, tanh, Tanh, [M, N, O]);
apply_impl!(Tensor4D, tanh, Tanh, [M, N, O, P]);

apply_impl!(Tensor0D, square, Square, []);
apply_impl!(Tensor1D, square, Square, [N]);
apply_impl!(Tensor2D, square, Square, [M, N]);
apply_impl!(Tensor3D, square, Square, [M, N, O]);
apply_impl!(Tensor4D, square, Square, [M, N, O, P]);

apply_impl!(Tensor0D, abs, Abs, []);
apply_impl!(Tensor1D, abs, Abs, [N]);
apply_impl!(Tensor2D, abs, Abs, [M, N]);
apply_impl!(Tensor3D, abs, Abs, [M, N, O]);
apply_impl!(Tensor4D, abs, Abs, [M, N, O, P]);

// macro_rules! neg_impl {
//     ($typename:ident, [$($const_names:tt),*]) => {
// impl<$(const $const_names: usize),*> std::ops::Neg for $typename<$($const_names, )* NoTape> {
//     type Output = Self;
//     fn neg(self) -> Self::Output {
//         Self::new(self.data().mapv(|v| -v))
//     }
// }

// impl<$(const $const_names: usize),*> std::ops::Neg for $typename<$($const_names, )* WithTape> {
//     type Output = Self;
//     fn neg(self) -> Self::Output {
//         let (t, tape) = self.without_tape();
//         let result = <Self as TensorWithTape>::NoTape::new_no_tape(t.data().mapv(|v| -v));
//         let tape = unary_op(tape, (&t, &result), t.data().mapv(|_| -1.0));
//         result.put_tape(tape)
//     }
// }
//     };
// }

// neg_impl!(Tensor0D, []);
// neg_impl!(Tensor1D, [N]);
// neg_impl!(Tensor2D, [M, N]);
// neg_impl!(Tensor3D, [M, N, O]);
// neg_impl!(Tensor4D, [M, N, O, P]);
