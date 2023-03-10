use crate::{shapes::*, tensor::*, tensor_ops::*};

use super::module::{Module, NonMutableModule, ZeroSizedModule};

macro_rules! activation_impls {
    ($struct_name:ident, $func_name:ident, #[$docstring:meta]) => {
        #[$docstring]
        #[derive(Default, Debug, Clone, Copy)]
        pub struct $struct_name;

        impl ZeroSizedModule for $struct_name {}
        impl NonMutableModule for $struct_name {}

        impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>>
            for $struct_name
        {
            type Output = Tensor<S, E, D, T>;
            type Error = D::Err;

            fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, D::Err> {
                input.$func_name()
            }
        }
    };
}

activation_impls!(ReLU, try_relu, #[doc="Unit struct that impls [Module] as calling [relu()] on `input`."]);
activation_impls!(GeLU, try_gelu, #[doc="Unit struct that impls [Module] as calling [gelu()] on `input`."]);
activation_impls!(Sin, try_sin, #[doc="Unit struct that impls [Module] as calling [sin()] on `input`."]);
activation_impls!(Cos, try_cos, #[doc="Unit struct that impls [Module] as calling [cos()] on `input`."]);
activation_impls!(Ln, try_ln, #[doc="Unit struct that impls [Module] as calling [ln()] on `input`."]);
activation_impls!(Exp, try_exp, #[doc="Unit struct that impls [Module] as calling [exp()] on `input`."]);
activation_impls!(Sigmoid, try_sigmoid, #[doc="Unit struct that impls [Module] as calling [sigmoid()] on `input`."]);
activation_impls!(Tanh, try_tanh, #[doc="Unit struct that impls [Module] as calling [tanh()] on `input`."]);
activation_impls!(Square, try_square, #[doc="Unit struct that impls [Module] as calling [square()] on `input`."]);
activation_impls!(Sqrt, try_sqrt, #[doc="Unit struct that impls [Module] as calling [sqrt()] on `input`."]);
activation_impls!(Abs, try_abs, #[doc="Unit struct that impls [Module] as calling [abs()] on `input`."]);

/// Unit struct that impls [Module] as calling [softmax()] on `input`."
#[derive(Default, Debug, Clone, Copy)]
pub struct Softmax;

impl ZeroSizedModule for Softmax {}
impl NonMutableModule for Softmax {}

impl<Ax: Axes, S, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for Softmax
where
    S: Shape<LastAxis = Ax> + ReduceShape<Ax>,
{
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, D::Err> {
        input.try_softmax::<Ax>()
    }
}

#[cfg(test)]
mod tests {
    use crate::{nn::*, tests::TestDevice};

    use super::*;

    #[test]
    fn test_nn_activations_relu() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = ReLU.forward_mut(t.clone());
        let r2 = relu(t);
        assert_eq!(r1.array(), r2.array());
    }

    #[test]
    fn test_nn_activations_gelu() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = GeLU.forward_mut(t.clone());
        let r2 = gelu(t);
        assert_eq!(r1.array(), r2.array());
    }

    #[test]
    fn test_nn_activations_sin() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Sin.forward_mut(t.clone());
        let r2 = sin(t);
        assert_eq!(r1.array(), r2.array());
    }
    #[test]
    fn test_nn_activations_cos() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Cos.forward_mut(t.clone());
        let r2 = cos(t);
        assert_eq!(r1.array(), r2.array());
    }
    #[test]
    fn test_nn_activations_ln() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([0.0, 1.0, 2.0, 3.0, 4.0]);
        let r1 = Ln.forward_mut(t.clone());
        let r2 = ln(t);
        assert_eq!(r1.array(), r2.array());
    }
    #[test]
    fn test_nn_activations_exp() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Exp.forward_mut(t.clone());
        let r2 = exp(t);
        assert_eq!(r1.array(), r2.array());
    }

    #[test]
    fn test_nn_activations_sigmoid() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Sigmoid.forward_mut(t.clone());
        let r2 = sigmoid(t);
        assert_eq!(r1.array(), r2.array());
    }
    #[test]
    fn test_nn_activations_tanh() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Tanh.forward_mut(t.clone());
        let r2 = tanh(t);
        assert_eq!(r1.array(), r2.array());
    }

    #[test]
    fn test_nn_activations_square() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Square.forward_mut(t.clone());
        let r2 = square(t);
        assert_eq!(r1.array(), r2.array());
    }

    #[test]
    fn test_nn_activations_sqrt() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([0.0, 1.0, 2.0, 3.0, 4.0]);
        let r1 = Sqrt.forward_mut(t.clone());
        let r2 = sqrt(t);
        assert_eq!(r1.array(), r2.array());
    }

    #[test]
    fn test_nn_activations_abs() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Abs.forward_mut(t.clone());
        let r2 = abs(t);
        assert_eq!(r1.array(), r2.array());
    }

    #[test]
    fn test_nn_activations_softmax() {
        let dev: TestDevice = Default::default();

        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Softmax.forward_mut(t.clone());
        let r2 = t.softmax();
        assert_eq!(r1.array(), r2.array());

        let t = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);
        let r1 = Softmax.forward_mut(t.clone());
        let r2 = t.softmax::<crate::shapes::Axis<1>>();
        assert_eq!(r1.array(), r2.array());
    }
}
