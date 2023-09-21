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

activation_impls!(ReLU, try_relu, #[doc="Calls [relu()]."]);
activation_impls!(FastGeLU, try_fast_gelu, #[doc="Calls [fast_gelu()]. This corresponds to `torch.nn.GELU(approximate='tanh')` in pytorch."]);
activation_impls!(
    AccurateGeLU,
    try_accurate_gelu, 
    #[doc=r#"Calls [accurate_gelu()]. The GeLU is defined as x * Phi(x) where Phi is the cumulative distribution function of a standard Normal Distribution. 
It is often implemented with a fast approximation using tanh (see [GeLU]). This corresponds to pytorch `torch.nn.GELU(approximate='none')` in pytorch."#]);
activation_impls!(Sin, try_sin, #[doc="Calls [sin()]."]);
activation_impls!(Cos, try_cos, #[doc="Calls [cos()]."]);
activation_impls!(Ln, try_ln, #[doc="Calls [ln()]."]);
activation_impls!(Exp, try_exp, #[doc="Calls [exp()]."]);
activation_impls!(Sigmoid, try_sigmoid, #[doc="Calls [sigmoid()]."]);
activation_impls!(Sigmoidr, try_sigmoidr, #[doc="Calls [sigmoidr()]."]);
activation_impls!(Tanh, try_tanh, #[doc="Calls [tanh()]."]);
activation_impls!(Square, try_square, #[doc="Calls [square()]."]);
activation_impls!(Sqrt, try_sqrt, #[doc="Calls [sqrt()]."]);
activation_impls!(Abs, try_abs, #[doc="Calls [abs()]."]);
activation_impls!(Softmax, try_softmax, #[doc="Calls [softmax()]."]);
activation_impls!(LogSoftmax, try_log_softmax, #[doc="Calls [log_softmax()]."]);

/// Use [FastGeLU] instead
#[deprecated(since = "0.12.0", note = "please use `FastGeLU` instead")]
#[derive(Default, Debug, Clone, Copy)]
pub struct GeLU;

#[allow(deprecated)]
impl ZeroSizedModule for GeLU {}
#[allow(deprecated)]
impl NonMutableModule for GeLU {}

#[allow(deprecated)]
impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for GeLU {
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, D::Err> {
        input.try_fast_gelu()
    }
}

/// Calls [prelu()] with constant value - defaults to 0.05
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU<E: Dtype>(pub E);

impl<E: Dtype> Default for LeakyReLU<E> {
    fn default() -> Self {
        Self(E::from_f32(0.05).unwrap())
    }
}

impl<E: Dtype> ZeroSizedModule for LeakyReLU<E> {}
impl<E: Dtype> NonMutableModule for LeakyReLU<E> {}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for LeakyReLU<E> {
    type Output = Tensor<S, E, D, T>;
    type Error = <Tensor<S, E, D, T> as HasErr>::Err;

    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        input.try_prelu(self.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::{nn::*, tests::TestDevice};

    #[allow(deprecated)]
    use super::GeLU;

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
    fn test_nn_activations_accurate_gelu() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = AccurateGeLU.forward_mut(t.clone());
        let r2 = accurate_gelu(t);
        assert_eq!(r1.array(), r2.array());
    }

    #[test]
    fn test_nn_activations_fast_gelu() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = FastGeLU.forward_mut(t.clone());
        #[allow(deprecated)]
        let r2 = GeLU.forward_mut(t.clone());
        let r3 = fast_gelu(t.clone());
        #[allow(deprecated)]
        let r4 = gelu(t);
        assert_eq!(r1.array(), r2.array());
        assert_eq!(r1.array(), r3.array());
        assert_eq!(r1.array(), r4.array());
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

    #[test]
    fn test_nn_activations_log_softmax() {
        let dev: TestDevice = Default::default();

        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = LogSoftmax.forward_mut(t.clone());
        let r2 = t.log_softmax();
        assert_eq!(r1.array(), r2.array());

        let t = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);
        let r1 = LogSoftmax.forward_mut(t.clone());
        let r2 = t.log_softmax::<crate::shapes::Axis<1>>();
        assert_eq!(r1.array(), r2.array());
    }

    #[test]
    fn test_nn_activations_leaky_relu() {
        let dev: TestDevice = Default::default();

        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = LeakyReLU(0.05).forward_mut(t.clone());
        let r2 = t.prelu(dev.tensor([0.05, 0.05, 0.05, 0.05, 0.05]));
        assert_eq!(r1.array(), r2.array());

        let t = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);
        let r1 = LeakyReLU(0.05).forward_mut(t.clone());
        let r2 = t.prelu(dev.tensor([[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]));
        assert_eq!(r1.array(), r2.array());
    }
}
