use crate::{arrays::*, gradients::Tape, optim::*, tensor::*, tensor_ops::*};

use super::module::{Module, ModuleMut, ResetParams};

macro_rules! activation_impls {
    ($struct_name:ident, $func_name:ident, #[$docstring:meta]) => {
        #[$docstring]
        #[derive(Default, Debug, Clone, Copy)]
        pub struct $struct_name;

        impl<E: Dtype, D: DeviceStorage> CanUpdateWithGradients<D, E> for $struct_name {
            /// Does nothing.
            fn update<U: UpdateParams<D, E>>(
                &mut self,
                _: &mut U,
                _: &mut UnusedTensors,
            ) -> Result<(), D::Err> {
                Ok(())
            }
        }

        impl ResetParams for $struct_name {
            /// Does nothing.
            fn reset_params(&mut self) {}
        }

        impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Module<Tensor<S, E, D, T>>
            for $struct_name
        {
            type Output = Tensor<S, E, D, T>;
            fn forward(&self, input: Tensor<S, E, D, T>) -> Self::Output {
                $func_name(input)
            }
        }

        impl<T> ModuleMut<T> for $struct_name
        where
            Self: Module<T>,
        {
            type Output = <Self as Module<T>>::Output;
            fn forward_mut(&mut self, input: T) -> Self::Output {
                self.forward(input)
            }
        }
    };
}

activation_impls!(ReLU, relu, #[doc="Unit struct that impls [Module] as calling [relu()] on `input`."]);
activation_impls!(Sin, sin, #[doc="Unit struct that impls [Module] as calling [sin()] on `input`."]);
activation_impls!(Cos, cos, #[doc="Unit struct that impls [Module] as calling [cos()] on `input`."]);
activation_impls!(Ln, ln, #[doc="Unit struct that impls [Module] as calling [ln()] on `input`."]);
activation_impls!(Exp, exp, #[doc="Unit struct that impls [Module] as calling [exp()] on `input`."]);
activation_impls!(Sigmoid, sigmoid, #[doc="Unit struct that impls [Module] as calling [sigmoid()] on `input`."]);
activation_impls!(Tanh, tanh, #[doc="Unit struct that impls [Module] as calling [tanh()] on `input`."]);
activation_impls!(Square, square, #[doc="Unit struct that impls [Module] as calling [square()] on `input`."]);
activation_impls!(Sqrt, sqrt, #[doc="Unit struct that impls [Module] as calling [sqrt()] on `input`."]);
activation_impls!(Abs, abs, #[doc="Unit struct that impls [Module] as calling [abs()] on `input`."]);

/// Unit struct that impls [Module] as calling [softmax()] on `input`."
#[derive(Default, Debug, Clone, Copy)]
pub struct Softmax;

impl<D: DeviceStorage, E: Dtype> CanUpdateWithGradients<D, E> for Softmax {
    /// Does nothing.
    fn update<U: UpdateParams<D, E>>(
        &mut self,
        _: &mut U,
        _: &mut UnusedTensors,
    ) -> Result<(), <D>::Err> {
        Ok(())
    }
}

impl ResetParams for Softmax {
    /// Does nothing.
    fn reset_params(&mut self) {}
}

impl<Ax: Axes, S: Shape<LastAxis = Ax> + ReduceShape<Ax>, E: Dtype, D: Device<E>, T: Tape<D>>
    Module<Tensor<S, E, D, T>> for Softmax
{
    type Output = Tensor<S, E, D, T>;
    fn forward(&self, input: Tensor<S, E, D, T>) -> Self::Output {
        input.softmax::<Ax>()
    }
}

impl<T> ModuleMut<T> for Softmax
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;
    fn forward_mut(&mut self, input: T) -> Self::Output {
        self.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::build_test_device;

    use super::*;

    #[test]
    fn test_nn_activations_relu() {
        let dev = build_test_device!();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = ReLU.forward_mut(t.clone());
        let r2 = relu(t);
        assert_eq!(r1.as_array(), r2.as_array());
    }

    #[test]
    fn test_nn_activations_sin() {
        let dev = build_test_device!();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Sin.forward_mut(t.clone());
        let r2 = sin(t);
        assert_eq!(r1.as_array(), r2.as_array());
    }
    #[test]
    fn test_nn_activations_cos() {
        let dev = build_test_device!();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Cos.forward_mut(t.clone());
        let r2 = cos(t);
        assert_eq!(r1.as_array(), r2.as_array());
    }
    #[test]
    fn test_nn_activations_ln() {
        let dev = build_test_device!();
        let t = dev.tensor([0.0, 1.0, 2.0, 3.0, 4.0]);
        let r1 = Ln.forward_mut(t.clone());
        let r2 = ln(t);
        assert_eq!(r1.as_array(), r2.as_array());
    }
    #[test]
    fn test_nn_activations_exp() {
        let dev = build_test_device!();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Exp.forward_mut(t.clone());
        let r2 = exp(t);
        assert_eq!(r1.as_array(), r2.as_array());
    }

    #[test]
    fn test_nn_activations_sigmoid() {
        let dev = build_test_device!();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Sigmoid.forward_mut(t.clone());
        let r2 = sigmoid(t);
        assert_eq!(r1.as_array(), r2.as_array());
    }
    #[test]
    fn test_nn_activations_tanh() {
        let dev = build_test_device!();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Tanh.forward_mut(t.clone());
        let r2 = tanh(t);
        assert_eq!(r1.as_array(), r2.as_array());
    }

    #[test]
    fn test_nn_activations_square() {
        let dev = build_test_device!();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Square.forward_mut(t.clone());
        let r2 = square(t);
        assert_eq!(r1.as_array(), r2.as_array());
    }

    #[test]
    fn test_nn_activations_sqrt() {
        let dev = build_test_device!();
        let t = dev.tensor([0.0, 1.0, 2.0, 3.0, 4.0]);
        let r1 = Sqrt.forward_mut(t.clone());
        let r2 = sqrt(t);
        assert_eq!(r1.as_array(), r2.as_array());
    }

    #[test]
    fn test_nn_activations_abs() {
        let dev = build_test_device!();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Abs.forward_mut(t.clone());
        let r2 = abs(t);
        assert_eq!(r1.as_array(), r2.as_array());
    }

    #[test]
    fn test_nn_activations_softmax() {
        let dev = build_test_device!();

        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Softmax.forward_mut(t.clone());
        let r2 = t.softmax();
        assert_eq!(r1.as_array(), r2.as_array());

        let t = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);
        let r1 = Softmax.forward_mut(t.clone());
        let r2 = t.softmax::<crate::arrays::Axis<1>>();
        assert_eq!(r1.as_array(), r2.as_array());
    }
}
