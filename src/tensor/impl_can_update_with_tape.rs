use super::*;
use crate::array_ops::SubElements;
use crate::prelude::GradientTape;

pub trait CanUpdateWithTape {
    fn update_with_tape(&mut self, tape: &GradientTape);
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H> CanUpdateWithTape for $typename<$($Vs, )* H> {
    fn update_with_tape(&mut self, tape: &GradientTape) {
        let gradient = tape.gradient(self);
        self.mut_data().sub_assign(gradient);
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
