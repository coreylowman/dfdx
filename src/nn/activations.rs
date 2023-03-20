use crate::{
    prelude::{BuildModule, BuildOnDevice, TensorCollection, TensorOptions},
    shapes::*,
    tensor::*,
    tensor_ops::*,
};

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
activation_impls!(GeLU, try_gelu, #[doc="Calls [gelu()]."]);
activation_impls!(Sin, try_sin, #[doc="Calls [sin()]."]);
activation_impls!(Cos, try_cos, #[doc="Calls [cos()]."]);
activation_impls!(Ln, try_ln, #[doc="Calls [ln()]."]);
activation_impls!(Exp, try_exp, #[doc="Calls [exp()]."]);
activation_impls!(Sigmoid, try_sigmoid, #[doc="Calls [sigmoid()]."]);
activation_impls!(Tanh, try_tanh, #[doc="Calls [tanh()]."]);
activation_impls!(Square, try_square, #[doc="Calls [square()]."]);
activation_impls!(Sqrt, try_sqrt, #[doc="Calls [sqrt()]."]);
activation_impls!(Abs, try_abs, #[doc="Calls [abs()]."]);

/// Calls [softmax()].
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

/// Calls [prelu()] with constant value.
#[derive(Default, Debug, Clone, Copy)]
pub struct LeakyReLU<E: Dtype>(E);

impl<E: Dtype> ZeroSizedModule for LeakyReLU<E> {}
impl<E: Dtype> NonMutableModule for LeakyReLU<E> {}

impl<Ax: Axes, S, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for LeakyReLU<E>
where
    S: Shape<LastAxis = Ax> + ReduceShape<Ax>,
    D: PReLUKernel<Tensor<S, E, D>, Tensor<(), E, D>, Output = Tensor<S, E, D>, Elem = E>,
{
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, D::Err> {
        let v = D::default().tensor_from_vec(vec![self.0], ());
        input.try_prelu(v)
    }
}

pub mod builder {
    use crate::prelude::Dtype;

    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub struct PReLU<E: Dtype>(E);
}

impl<E: Dtype, D: Device<E>> BuildOnDevice<D, E> for builder::PReLU<E>
where
    PReLU<E, D>: BuildModule<D, E>,
{
    type Built = PReLU<E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

/// Calls [prelu()] with learnable value.
#[derive(Debug, Clone)]
pub struct PReLU<E: Dtype, D: Device<E>> {
    a: Tensor<(), E, D>,
}

impl<E: Dtype, D: Device<E>> Default for PReLU<E, D> {
    fn default() -> Self {
        let dev = D::default();
        Self {
            a: dev.tensor(E::from_f32(0.05).unwrap()),
        }
    }
}

impl<E: Dtype, D: Device<E>> From<E> for PReLU<E, D> {
    fn from(value: E) -> Self {
        let dev = D::default();
        Self {
            a: dev.tensor(value),
        }
    }
}

impl<E: Dtype, D: Device<E>> NonMutableModule for PReLU<E, D> {}

impl<Ax: Axes, S, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for PReLU<E, D>
where
    S: Shape<LastAxis = Ax> + ReduceShape<Ax>,
    D: PReLUKernel<Tensor<S, E, D>, Tensor<(), E, D>, Output = Tensor<S, E, D>, Elem = E>,
{
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, D::Err> {
        input.try_prelu(self.a.clone())
    }
}

impl<E: Dtype, D: Device<E>> TensorCollection<E, D> for PReLU<E, D> {
    type To<E2: Dtype, D2: Device<E2>> = PReLU<E2, D2>;

    fn iter_tensors<V: crate::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_tensor("a", |p| &p.a, |p| &mut p.a, TensorOptions::reset_to_zeros())?;
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        nn::*,
        tests::{assert_close, TestDevice},
    };

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

    #[test]
    fn test_nn_activations_leaky_relu() {
        let dev: TestDevice = Default::default();

        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = LeakyReLU(0.05).forward_mut(t.clone());
        let r2 = t.prelu(dev.tensor(0.05));
        assert_eq!(r1.array(), r2.array());

        let t = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);
        let r1 = LeakyReLU(0.05).forward_mut(t.clone());
        let r2 = t.prelu(dev.tensor(0.05));
        assert_eq!(r1.array(), r2.array());
    }

    #[test]
    fn test_nn_activations_prelu() {
        let dev: TestDevice = Default::default();

        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = PReLU::from(0.05).forward_mut(t.clone());
        let r2 = t.prelu(dev.tensor(0.05));
        assert_eq!(r1.array(), r2.array());

        let t = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);
        let r1 = PReLU::from(0.05).forward_mut(t.clone());
        let r2 = t.prelu(dev.tensor(0.05));
        assert_eq!(r1.array(), r2.array());

        let model = (Tanh, PReLU::from(0.05));
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let out = model.forward(t);
        assert_close(
            &out.array(),
            &[-0.04820138, -0.03807970, 0.0, 0.76159415, 0.96402758],
        )
    }
}
