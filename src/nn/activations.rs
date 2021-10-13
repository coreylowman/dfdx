use super::module::{Init, Module};
use crate::gradients::{GradientTape, Taped};
use crate::tensor::Tensor;
use ndarray_rand::rand::Rng;

macro_rules! nn_activation {
    ($module_name:ident, $act_fn:ident) => {
        #[derive(Debug, Default)]
        pub struct $module_name;

        impl Init for $module_name {
            fn init<R: Rng>(&mut self, _rng: &mut R) {}
        }

        impl Taped for $module_name {
            fn update(&mut self, _tape: &GradientTape) {}
        }

        impl<T: Tensor> Module<T, T> for $module_name {
            fn forward(&mut self, input: &mut T) -> T {
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
