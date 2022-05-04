use crate::prelude::*;

pub fn apply<T: Tensor, F: DifferentiableFunction>(t: T) -> T {
    let result = T::NoTape::new(t.data().mapv_elems(F::f));
    let (t, mut tape_holder) = t.split_tape_holder();
    let deriv: T::ArrayType = t.data().mapv_elems(F::df);
    let _t = t.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let d_grad = deriv.mul(tape.gradient(&_result));
        tape.mut_gradient(&_t).add_assign(&d_grad);
    });
    result.with_tape_holder(tape_holder)
}

macro_rules! apply_impl {
    ($trait_name:ident, $method_name:ident, $activation_struct:ident) => {
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

pub fn apply_ref<T, F: DifferentiableFunction>(t: &T) -> T
where
    T: Tensor<TapeHolder = NoTape> + TensorCreator,
{
    T::new(t.data().mapv_elems(F::f))
}

macro_rules! apply_ref_impl {
    ($trait_name:ident, $method_name:ident, $activation_struct:ident) => {
        pub trait $trait_name {
            fn $method_name(&self) -> Self;
        }

        impl<T: Tensor<TapeHolder = NoTape> + TensorCreator> $trait_name for T {
            fn $method_name(&self) -> Self {
                apply_ref::<Self, $activation_struct>(self)
            }
        }
    };
}

apply_ref_impl!(HasReLURefMethod, relu_, ReLU);
apply_ref_impl!(HasSinRefMethod, sin_, Sin);
apply_ref_impl!(HasCosRefMethod, cos_, Cos);
apply_ref_impl!(HasLnRefMethod, ln_, Ln);
apply_ref_impl!(HasExpRefMethod, exp_, Exp);
apply_ref_impl!(HasSigmoidRefMethod, sigmoid_, Sigmoid);
apply_ref_impl!(HasTanhRefMethod, tanh_, Tanh);
apply_ref_impl!(HasSquareRefMethod, square_, Square);
apply_ref_impl!(HasAbsRefMethod, abs_, Abs);
