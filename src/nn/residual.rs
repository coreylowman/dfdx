use crate::{optim::*, shapes::*, tensor::SplitTape, tensor_ops::Device};

use super::{Module, ModuleMut, ResetParams};

/// A residual connection around `F`: `F(x) + x`,
/// as introduced in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
///
/// # Generics
/// - `F`: The underlying module to do a skip connection around.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let module: Residual<ReLU> = dev.build_module();
/// let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let y = module.forward(x);
/// assert_eq!(y.array(), [-2.0, -1.0, 0.0, 2.0, 4.0]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Residual<F>(pub F);

impl<D: Device<E>, E: Dtype, F: GradientUpdate<D, E>> GradientUpdate<D, E> for Residual<F> {
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), D::Err>
    where
        U: ParamUpdater<D, E>,
    {
        self.0.update(updater, unused)
    }
}

impl<D: Device<E>, E: Dtype, F: ResetParams<D, E>> ResetParams<D, E> for Residual<F> {
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        Ok(Self(ResetParams::try_build(device)?))
    }
    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        self.0.try_reset_params()
    }
}

impl<T: SplitTape + std::ops::Add<T, Output = T>, F: Module<T, Output = T>> Module<T>
    for Residual<F>
{
    type Output = T;
    fn forward(&self, x: T) -> Self::Output {
        self.0.forward(x.with_empty_tape()) + x
    }
}

impl<T: SplitTape + std::ops::Add<T, Output = T>, F: ModuleMut<T, Output = T>> ModuleMut<T>
    for Residual<F>
{
    type Output = T;
    fn forward_mut(&mut self, x: T) -> Self::Output {
        self.0.forward_mut(x.with_empty_tape()) + x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{assert_close, TestDevice};
    use crate::{
        nn::{Linear, ModuleBuilder},
        tensor::*,
        tensor_ops::*,
    };

    #[test]
    fn test_residual_reset() {
        let dev: TestDevice = Default::default();
        let model: Residual<Linear<2, 5, _>> = dev.build_module();
        assert_ne!(model.0.weight.array(), [[0.0; 2]; 5]);
        assert_ne!(model.0.bias.array(), [0.0; 5]);
    }

    #[test]
    fn test_residual_gradients() {
        let dev: TestDevice = Default::default();

        let model: Residual<Linear<2, 2, _>> = dev.build_module();

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
