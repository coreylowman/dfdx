use crate::{
    shapes::*,
    tensor::*,
    tensor_ops::{Device, TryAdd},
};

use super::*;

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
/// type Model = GeneralizedResidual<ReLU, Square>;
/// let model = dev.build_module::<Model, f32>();
/// let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let y = model.forward(x);
/// assert_eq!(y.array(), [4.0, 1.0, 0.0, 2.0, 6.0]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct GeneralizedResidual<F, R> {
    pub f: F,
    pub r: R,
}

impl<D: Device<E>, E: Dtype, F: BuildOnDevice<D, E>, R: BuildOnDevice<D, E>> BuildOnDevice<D, E>
    for GeneralizedResidual<F, R>
{
    type Built = GeneralizedResidual<F::Built, R::Built>;
}

impl<E: Dtype, D: Device<E>, F: TensorCollection<E, D>, R: TensorCollection<E, D>>
    TensorCollection<E, D> for GeneralizedResidual<F, R>
{
    type To<E2: Dtype, D2: Device<E2>> = GeneralizedResidual<F::To<E2, D2>, R::To<E2, D2>>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("f", |s| &s.f, |s| &mut s.f),
                Self::module("r", |s| &s.r, |s| &mut s.r),
            ),
            |(f, r)| GeneralizedResidual { f, r },
        )
    }
}

impl<T: WithEmptyTape, F: Module<T>, R: Module<T, Output = F::Output, Error = F::Error>> Module<T>
    for GeneralizedResidual<F, R>
where
    F::Output: TryAdd<F::Output> + HasErr<Err = F::Error>,
{
    type Output = F::Output;
    type Error = F::Error;

    fn try_forward(&self, x: T) -> Result<Self::Output, F::Error> {
        self.f
            .try_forward(x.with_empty_tape())?
            .try_add(self.r.try_forward(x)?)
    }
}

impl<T: WithEmptyTape, F: ModuleMut<T>, R: ModuleMut<T, Output = F::Output, Error = F::Error>>
    ModuleMut<T> for GeneralizedResidual<F, R>
where
    F::Output: TryAdd<F::Output> + HasErr<Err = F::Error>,
{
    type Output = F::Output;
    type Error = F::Error;

    fn try_forward_mut(&mut self, x: T) -> Result<Self::Output, F::Error> {
        self.f
            .try_forward_mut(x.with_empty_tape())?
            .try_add(self.r.try_forward_mut(x)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::builders::{DeviceBuildExt, Linear};
    use crate::{tensor_ops::*, tests::*};

    #[test]
    fn test_reset_generalized_residual() {
        let dev: TestDevice = Default::default();

        type Model = GeneralizedResidual<Linear<2, 5>, Linear<2, 5>>;
        let model = dev.build_module::<Model, f32>();
        assert_ne!(model.f.weight.array(), [[0.0; 2]; 5]);
        assert_ne!(model.f.bias.array(), [0.0; 5]);
        assert_ne!(model.r.weight.array(), [[0.0; 2]; 5]);
        assert_ne!(model.r.bias.array(), [0.0; 5]);
    }

    #[test]
    fn test_generalized_residual_gradients() {
        let dev: TestDevice = Default::default();

        type Model = GeneralizedResidual<Linear<2, 2>, Linear<2, 2>>;
        let model = dev.build_module::<Model, f32>();

        let x = dev.sample_normal::<Rank2<4, 2>>();
        let y = model.forward(x.leaky_trace());

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
