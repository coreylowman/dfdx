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
        impl<F: DifferentiableFunction + Default $(, const $const_names: usize)*> Module<$typename<$($const_names, )* NoTape>> for F {
            type Output = $typename<$($const_names, )* NoTape>;
            fn forward(&self, input: $typename<$($const_names, )* NoTape>) -> Self::Output {
                apply_no_tape::<$typename<$($const_names, )* NoTape>, Self>(input)
            }
        }

        impl<F: DifferentiableFunction + Default $(, const $const_names: usize)*> Module<$typename<$($const_names, )* WithTape>> for F {
            type Output = $typename<$($const_names, )* WithTape>;
            fn forward(&self, input: $typename<$($const_names, )* WithTape>) -> Self::Output {
                apply_with_tape::<$typename<$($const_names, )* WithTape>, Self>(input)
            }
        }
    };
}

activation_impl!(Tensor0D, []);
activation_impl!(Tensor1D, [N]);
activation_impl!(Tensor2D, [M, N]);
activation_impl!(Tensor3D, [M, N, O]);
activation_impl!(Tensor4D, [M, N, O, P]);
