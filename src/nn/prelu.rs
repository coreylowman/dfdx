use crate::{
    prelude::{
        BuildModule, BuildOnDevice, Module, NonMutableModule, TensorCollection, TensorOptions,
    },
    shapes::*,
    tensor::*,
    tensor_ops::*,
};

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

impl<S: ConstShape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>>
    for PReLU<E, D>
where
    Tensor<S, E, D, T>: TryPReLU<Tensor<S, E, D, NoneTape>>,
{
    type Output = Tensor<S, E, D, T>;
    type Error = <Tensor<S, E, D, T> as HasErr>::Err;

    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        input.try_prelu(self.a.retaped().broadcast())
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
where
    Tensor<(C,), E, D, T>: TryPReLU<Tensor<(C,), E, D, NoneTape>>,
{
    type Output = Tensor<(C,), E, D, T>;

    type Error = <Tensor<(C,), E, D, T> as HasErr>::Err;

    fn try_forward(&self, input: Tensor<(C,), E, D, T>) -> Result<Self::Output, Self::Error> {
        input.try_prelu(self.a.retaped())
    }
}

macro_rules! prelu1d {
    (($($InDims:tt),*), $Axes:ty) => {
        impl<E: Dtype, D: Device<E>, T: Tape<E, D> + Merge<NoneTape>, $($InDims: ConstDim),*> Module<Tensor<($($InDims),*), E, D, T>> for PReLU1D<C,E, D>
        where ($($InDims),*): ReduceShapeTo<(C,), $Axes>,
        Tensor<($($InDims),*), E, D, T>: TryPReLU<Tensor<($($InDims),*), E, D, NoneTape>>,
        {
            type Output = Tensor<($($InDims),*), E, D, T>;
            type Error = <Tensor<($($InDims),*), E, D, T> as HasErr>::Err;

            fn try_forward(&self, input: Tensor<($($InDims),*), E, D, T>) -> Result<Self::Output, Self::Error> {
                input.try_prelu(self.a.retaped().broadcast())
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
    use modules::Tanh;

    use crate::{nn::*, tests::*};

    use super::*;

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
        assert_close_to_literal!(out, [-0.04820138, -0.03807970, 0.0, 0.76159415, 0.96402758])
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
        assert_close_to_literal!(out, [-0.04820138, -0.03807970, 0.0, 0.76159415, 0.96402758])
    }
}
