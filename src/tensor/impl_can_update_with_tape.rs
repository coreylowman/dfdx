use super::*;
use crate::array_ops::SubElements;
use crate::prelude::Gradients;

pub trait CanUpdateWithGradients {
    fn update_with_grads(&mut self, grads: &Gradients);
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H> CanUpdateWithGradients for $typename<$($Vs, )* H> {
    fn update_with_grads(&mut self, grads: &Gradients) {
        let gradient = grads.gradient(self);
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
