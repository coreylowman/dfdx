use crate::prelude::*;
use rand::{distributions::Distribution, Rng};

macro_rules! activation_impls {
    ($typename:ty) => {
        impl CanUpdateWithGradients for $typename {
            fn update<G: GradientProvider>(&mut self, _: &mut G) {}
        }

        impl Randomize for $typename {
            fn randomize<R: Rng, D: Distribution<f32>>(&mut self, _: &mut R, _: &D) {}
        }

        impl<T: Tensor> Module<T> for $typename {
            type Output = T;
            fn forward(&self, input: T) -> Self::Output {
                apply::<T, Self>(input)
            }
        }
    };
}

activation_impls!(ReLU);
activation_impls!(Sin);
activation_impls!(Cos);
activation_impls!(Ln);
activation_impls!(Exp);
activation_impls!(Sigmoid);
activation_impls!(Tanh);
activation_impls!(Square);
activation_impls!(Abs);
