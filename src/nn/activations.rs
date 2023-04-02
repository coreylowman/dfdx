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

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for Softmax {
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, D::Err> {
        input.try_softmax::<S::LastAxis>()
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

pub mod builder {
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub struct PReLU;

    use core::marker::PhantomData;

    use crate::prelude::ConstDim;

    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub struct PReLU1D<C: ConstDim>(PhantomData<C>);
}

impl<E: Dtype, D: Device<E>> BuildOnDevice<D, E> for builder::PReLU
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
    pub a: Tensor<(), E, D>,
}

impl<E: Dtype, D: Device<E>> NonMutableModule for PReLU<E, D> {}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for PReLU<E, D> {
    type Output = Tensor<S, E, D, T>;
    type Error = <Tensor<S, E, D, T> as HasErr>::Err;

    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        let scale = self.a.retaped::<T>().try_broadcast_like(&input.shape)?;
        input.try_prelu(scale)
    }
}

impl<E: Dtype, D: Device<E>> TensorCollection<E, D> for PReLU<E, D> {
    type To<E2: Dtype, D2: Device<E2>> = PReLU<E2, D2>;

    fn iter_tensors<V: crate::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            Self::tensor("a", |p| &p.a, |p| &mut p.a, TensorOptions::reset_to_zeros()),
            |a| PReLU { a },
        )
    }
}

/// Calls [prelu()] with learnable values along second dimension.
#[derive(Debug, Clone)]
pub struct PReLU1D<C: ConstDim, E: Dtype, D: Device<E>> {
    pub a: Tensor<(C,), E, D>,
}

impl<C: ConstDim, E: Dtype, D: Device<E>> BuildOnDevice<D, E> for builder::PReLU1D<C>
where
    PReLU1D<C, E, D>: BuildModule<D, E>,
{
    type Built = PReLU1D<C, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

impl<C: ConstDim, E: Dtype, D: Device<E>> NonMutableModule for PReLU1D<C, E, D> {}

impl<C: ConstDim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(C,), E, D, T>>
    for PReLU1D<C, E, D>
{
    type Output = Tensor<(C,), E, D, T>;
    type Error = <Tensor<(C,), E, D, T> as HasErr>::Err;

    fn try_forward(&self, input: Tensor<(C,), E, D, T>) -> Result<Self::Output, Self::Error> {
        input.try_prelu(self.a.retaped::<T>())
    }
}

macro_rules! prelu1d {
    (($($InDims:tt),*), $Axes:ty) => {
        impl<E: Dtype, D: Device<E>, T: Tape<E, D>, $($InDims: ConstDim),*> Module<Tensor<($($InDims),*), E, D, T>> for PReLU1D<C,E, D>
        where ($($InDims),*): ReduceShapeTo<(C,), $Axes>,
        {
            type Output = Tensor<($($InDims),*), E, D, T>;
            type Error = <Tensor<($($InDims),*), E, D, T> as HasErr>::Err;

            fn try_forward(&self, input: Tensor<($($InDims),*), E, D, T>) -> Result<Self::Output, Self::Error> {
                input.try_prelu(self.a.retaped::<T>().broadcast())
            }
        }
    };
}

prelu1d!((B, C), Axis<0>);
prelu1d!((B, C, M), Axes2<0, 2>);
prelu1d!((B, C, M, N), Axes3<0, 2, 3>);

impl<C: ConstDim, E: Dtype, D: Device<E>> TensorCollection<E, D> for PReLU1D<C, E, D> {
    type To<E2: Dtype, D2: Device<E2>> = PReLU1D<C, E2, D2>;

    fn iter_tensors<V: crate::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            Self::tensor("a", |p| &p.a, |p| &mut p.a, TensorOptions::reset_to_zeros()),
            |a| PReLU1D { a },
        )
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
        let r2 = t.prelu(dev.tensor([0.05, 0.05, 0.05, 0.05, 0.05]));
        assert_eq!(r1.array(), r2.array());

        let t = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);
        let r1 = LeakyReLU(0.05).forward_mut(t.clone());
        let r2 = t.prelu(dev.tensor([[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]));
        assert_eq!(r1.array(), r2.array());
    }

    #[test]
    fn test_nn_activations_prelu() {
        let dev: TestDevice = Default::default();

        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = PReLU {
            a: dev.tensor(0.05),
        }
        .forward_mut(t.clone());
        let r2 = t.prelu(dev.tensor([0.05, 0.05, 0.05, 0.05, 0.05]));
        assert_eq!(r1.array(), r2.array());

        let t = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);
        let r1 = PReLU {
            a: dev.tensor(0.05),
        }
        .forward_mut(t.clone());
        let r2 = t.prelu(dev.tensor([[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]));
        assert_eq!(r1.array(), r2.array());

        let model = (
            Tanh,
            PReLU {
                a: dev.tensor(0.05),
            },
        );
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let out = model.forward(t);
        assert_close(
            &out.array(),
            &[-0.04820138, -0.03807970, 0.0, 0.76159415, 0.96402758],
        )
    }

    #[test]
    fn test_nn_activations_prelu_1d() {
        let dev: TestDevice = Default::default();

        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = PReLU1D {
            a: dev.tensor([0.05, 0.06, 0.07, 0.08, 0.09]),
        }
        .forward_mut(t.clone());
        let r2 = t.prelu(dev.tensor([0.05, 0.06, 0.07, 0.08, 0.09]));
        assert_eq!(r1.array(), r2.array());

        let t = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);
        let r1 = PReLU1D {
            a: dev.tensor([0.05, 0.07, 0.09]),
        }
        .forward_mut(t.clone());
        let r2 = t.prelu(dev.tensor([[0.05, 0.07, 0.09], [0.05, 0.07, 0.09]]));
        assert_eq!(r1.array(), r2.array());

        let mut model = dev.build_module::<(Tanh, builder::PReLU1D<Const<5>>), f32>();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        model.1.a = dev.tensor([0.05, 0.05, 0.05, 0.05, 0.05]);
        let out = model.forward(t);
        assert_close(
            &out.array(),
            &[-0.04820138, -0.0380797, 0.0, 0.7615941, 0.9640275],
        )
    }
}
