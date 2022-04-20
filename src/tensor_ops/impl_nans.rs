use super::add_unary_op;
use crate::prelude::*;

fn nans_to<T: Tensor>(t: T, value: f32) -> T {
    let result = T::NoTape::new(t.data().mapv(|v| if v.is_nan() { value } else { v }));
    let (t, mut tape_holder) = t.split_tape_holder();
    tape_holder.update_with(|tape| {
        let deriv = t.data().mapv(|v| if v.is_nan() { 0.0 } else { 1.0 });
        add_unary_op(tape, (&t, &result), deriv)
    });
    result.with_tape_holder(tape_holder)
}

pub trait CanReplaceNans {
    fn nans_to(self, value: f32) -> Self;
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> CanReplaceNans for $typename<$($Vs, )* H> {
    fn nans_to(self, value: f32) -> Self {
        nans_to(self, value)
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
