use crate::{optim::*, shapes::*, tensor::*, tensor_ops::*};

use super::{Module, ModuleMut, ResetParams};

/// A residual connection `R` around `F`: `F(x) + R(x)`,
/// as introduced in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
///
/// # Generics
/// - `F`: The underlying module to do a skip connection around.
/// - `R`: The underlying residual module
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let module: GeneralizedResidual<ReLU, Square> = dev.build_module();
/// let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let y = module.forward(x);
/// assert_eq!(y.array(), [4.0, 1.0, 0.0, 2.0, 6.0]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct GeneralizedResidual<F, R> {
    pub f: F,
    pub r: R,
}

impl<D: Device<E>, E: Dtype, F: GradientUpdate<D, E>, R: GradientUpdate<D, E>> GradientUpdate<D, E>
    for GeneralizedResidual<F, R>
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), <D>::Err>
    where
        U: ParamUpdater<D, E>,
    {
        self.f.update(updater, unused)?;
        self.r.update(updater, unused)?;
        Ok(())
    }
}

impl<D: Device<E>, E: Dtype, F: ResetParams<D, E>, R: ResetParams<D, E>> ResetParams<D, E>
    for GeneralizedResidual<F, R>
{
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        Ok(Self {
            f: ResetParams::try_build(device)?,
            r: ResetParams::try_build(device)?,
        })
    }
    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        self.f.try_reset_params()?;
        self.r.try_reset_params()?;
        Ok(())
    }
}

impl<T: SplitTape, F: Module<T>, R: Module<T, Output = F::Output>> Module<T>
    for GeneralizedResidual<F, R>
where
    F::Output: std::ops::Add<F::Output>,
{
    type Output = <F::Output as std::ops::Add<F::Output>>::Output;
    fn forward(&self, x: T) -> Self::Output {
        self.f.forward(x.with_empty_tape()) + self.r.forward(x)
    }
}

impl<T: SplitTape, F: ModuleMut<T>, R: ModuleMut<T, Output = F::Output>> ModuleMut<T>
    for GeneralizedResidual<F, R>
where
    F::Output: std::ops::Add<F::Output>,
{
    type Output = <F::Output as std::ops::Add<F::Output>>::Output;
    fn forward_mut(&mut self, x: T) -> Self::Output {
        self.f.forward_mut(x.with_empty_tape()) + self.r.forward_mut(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, ModuleBuilder};
    use crate::tests::{assert_close, TestDevice};

    #[test]
    fn test_reset_generalized_residual() {
        let dev: TestDevice = Default::default();

        let model: GeneralizedResidual<Linear<2, 5, _>, Linear<2, 5, _>> = dev.build_module();
        assert_ne!(model.f.weight.array(), [[0.0; 2]; 5]);
        assert_ne!(model.f.bias.array(), [0.0; 5]);
        assert_ne!(model.r.weight.array(), [[0.0; 2]; 5]);
        assert_ne!(model.r.bias.array(), [0.0; 5]);
    }

    #[test]
    fn test_generalized_residual_gradients() {
        let dev: TestDevice = Default::default();

        let model: GeneralizedResidual<Linear<2, 2, _>, Linear<2, 2, _>> = dev.build_module();

        let x = dev.randn::<Rank2<4, 2>>();
        let y = model.forward(x.trace());

        #[rustfmt::skip]
        assert_close(&y.array(), &[[-0.81360567, -1.1473482], [1.0925694, 0.17383915], [-0.32519114, 0.49806428], [0.08259219, -0.7277866]]);

        let g = y.mean().backward();
        assert_close(&g.get(&x).array(), &[[0.15889636, 0.062031522]; 4]);
        assert_close(&g.get(&model.f.weight).array(), &[[-0.025407, 0.155879]; 2]);
        assert_close(&g.get(&model.f.bias).array(), &[0.5; 2]);
        assert_close(&g.get(&model.r.weight).array(), &[[-0.025407, 0.155879]; 2]);
        assert_close(&g.get(&model.r.bias).array(), &[0.5; 2]);
    }
}
