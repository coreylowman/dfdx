use super::traits::Module;
use crate::prelude::*;
use rand::{distributions::Distribution, Rng};

impl<F: DifferentiableFunction> CanUpdateWithTape for F {
    fn update_with_tape(&mut self, _: &GradientTape) {}
}

impl<F: DifferentiableFunction> Randomize for F {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, _: &mut R, _: &D) {}
}

macro_rules! activation_impl {
    ($typename:ident, [$($const_names:tt),*]) => {
impl<F: DifferentiableFunction + Default, H: TapeHolder $(, const $const_names: usize)*> Module<$typename<$($const_names, )* H>> for F {
    type Output = $typename<$($const_names, )* H>;
    fn forward(&self, input: $typename<$($const_names, )* H>) -> Self::Output {
        apply::<$typename<$($const_names, )* H>, Self>(input)
    }
}
    };
}

activation_impl!(Tensor0D, []);
activation_impl!(Tensor1D, [N]);
activation_impl!(Tensor2D, [M, N]);
activation_impl!(Tensor3D, [M, N, O]);
activation_impl!(Tensor4D, [M, N, O, P]);
