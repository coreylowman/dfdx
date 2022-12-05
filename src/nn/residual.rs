use crate::{gradients::Tape, optim::*, shapes::*, tensor::Tensor, tensor_ops::Device};

use super::{BuildModule, Module, ModuleMut};

/// A residual connection around `F`: `F(x) + x`,
/// as introduced in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
///
/// # Generics
/// - `F`: The underlying module to do a skip connection around.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// let module: Residual<ReLU> = Default::default();
/// let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let y = module.forward(x);
/// assert_eq!(y.data(), &[-2.0, -1.0, 0.0, 2.0, 4.0]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Residual<F>(pub F);

impl<D: Device<E>, E: Dtype, F: CanUpdateWithGradients<D, E>> CanUpdateWithGradients<D, E>
    for Residual<F>
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), D::Err>
    where
        U: UpdateParams<D, E>,
    {
        self.0.update(updater, unused)
    }
}

impl<D: Device<E>, E: Dtype, F: BuildModule<D, E>> BuildModule<D, E> for Residual<F> {
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        Ok(Self(F::try_build(device)?))
    }
    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        self.0.try_reset_params()
    }
}

impl<
        S: Shape,
        E: Dtype,
        D: Device<E>,
        T: Tape<D>,
        F: Module<Tensor<S, E, D, T>, Output = Tensor<S, E, D, T>>,
    > Module<Tensor<S, E, D, T>> for Residual<F>
{
    type Output = Tensor<S, E, D, T>;
    fn forward(&self, x: Tensor<S, E, D, T>) -> Self::Output {
        self.0.forward(x.retaped::<T>()) + x
    }
}

impl<
        S: Shape,
        E: Dtype,
        D: Device<E>,
        T: Tape<D>,
        F: ModuleMut<Tensor<S, E, D, T>, Output = Tensor<S, E, D, T>>,
    > ModuleMut<Tensor<S, E, D, T>> for Residual<F>
{
    type Output = Tensor<S, E, D, T>;
    fn forward_mut(&mut self, x: Tensor<S, E, D, T>) -> Self::Output {
        self.0.forward_mut(x.retaped::<T>()) + x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{assert_close, build_test_device};
    use crate::{nn::Linear, tensor::*, tensor_ops::*};

    #[test]
    fn test_residual_reset() {
        let dev = build_test_device!();
        let model: Residual<Linear<2, 5, _>> = BuildModule::build(&dev);
        assert_ne!(model.0.weight.array(), [[0.0; 2]; 5]);
        assert_ne!(model.0.bias.array(), [0.0; 5]);
    }

    #[test]
    fn test_residual_gradients() {
        let dev = build_test_device!();

        let model: Residual<Linear<2, 2, _>> = BuildModule::build(&dev);

        let x = dev.randn::<Rank2<4, 2>>();
        let y = model.forward(x.trace());

        #[rustfmt::skip]
        assert_close(&y.array(), &[[0.25372928, -2.4258814],[1.7892148, -2.6242268],[1.5131638, 0.23407778],[3.4201493, 1.597525]]);

        let g = y.mean().backward();
        assert_close(&g.get(&model.0.weight).array(), &[[0.475242, -0.075136]; 2]);
        assert_close(&g.get(&model.0.bias).array(), &[0.5; 2]);
        assert_close(&g.get(&x).array(), &[[0.18806472, 0.21419683]; 4]);
    }
}
