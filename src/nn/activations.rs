use super::traits::{Init, Module};
use crate::{
    gradients::{traits::Taped, GradientTape},
    tensor::traits::{Activations, Batch, Tensor},
};
use ndarray_rand::rand::Rng;
use std::marker::PhantomData;

macro_rules! nn_activation {
    ($module_name:ident, $act_fn:ident) => {
        #[derive(Debug, Default)]
        pub struct $module_name<T: Tensor + Batch>(PhantomData<T>);

        impl<T: Tensor + Batch> Init for $module_name<T> {
            fn init<R: Rng>(&mut self, _rng: &mut R) {}
        }

        impl<T: Tensor + Batch> Taped for $module_name<T> {
            fn update(&mut self, _tape: &GradientTape) {}
        }

        impl<T: Tensor + Batch> Module for $module_name<T> {
            type Input = T;
            type Output = T;

            fn forward<const B: usize>(&mut self, input: &mut T::Batched<B>) -> T::Batched<B> {
                input.$act_fn()
            }
        }
    };
}

nn_activation!(ReLU, relu);
nn_activation!(Sin, sin);
nn_activation!(Cos, cos);
nn_activation!(Sigmoid, sigmoid);
nn_activation!(Tanh, tanh);
