use super::ops::add_unary_op;
use crate::prelude::*;

pub fn value_mask<T: Tensor>(t: T, other: &T::NoTape, value: f32) -> T {
    let result = T::NoTape::new(
        ndarray::Zip::from(t.data())
            .and(other.data())
            .map_collect(|&a, b| if b.eq(&value) { value } else { a }),
    );
    let (t, mut tape_holder) = t.split_tape_holder();
    tape_holder.update_with(|tape| {
        let deriv = other.data().map(|v| if v.eq(&value) { 0.0 } else { 1.0 });
        add_unary_op(tape, (&t, &result), deriv)
    });
    result.with_tape_holder(tape_holder)
}

pub trait CanMaskValues<Mask> {
    fn value_mask(self, mask: Mask, value: f32) -> Self;
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> CanMaskValues<&$typename<$($Vs, )* NoTape>> for $typename<$($Vs, )* H> {
    fn value_mask(self, mask: &$typename<$($Vs, )* NoTape>, value: f32) -> Self {
        value_mask(self, mask, value)
    }
}
    };
}

tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    #[test]
    fn test_mask_1d() {
        todo!();
    }

    #[test]
    fn test_mask_2d() {
        todo!();
    }
}
