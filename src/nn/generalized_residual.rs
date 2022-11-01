use crate::gradients::{GradientProvider, UnusedTensors};
use crate::prelude::*;

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
/// let module: GeneralizedResidual<ReLU, Square> = Default::default();
/// let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let y = module.forward(x);
/// assert_eq!(y.data(), &[4.0, 1.0, 0.0, 2.0, 6.0]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct GeneralizedResidual<F, R> {
    pub f: F,
    pub r: R,
}

impl<F: CanUpdateWithGradients, R: CanUpdateWithGradients> CanUpdateWithGradients
    for GeneralizedResidual<F, R>
{
    /// Pass through to `F`'s [CanUpdateWithGradients].
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.f.update(grads, unused);
        self.r.update(grads, unused);
    }
}

impl<F: ResetParams, R: ResetParams> ResetParams for GeneralizedResidual<F, R> {
    /// Pass through to `F`'s [ResetParams].
    fn reset_params<RNG: rand::Rng>(&mut self, rng: &mut RNG) {
        self.f.reset_params(rng);
        self.r.reset_params(rng);
    }
}

impl<F, R, T> Module<T> for GeneralizedResidual<F, R>
where
    T: Tensor<Dtype = f32>,
    F: Module<T>,
    R: Module<T, Output = F::Output>,
    F::Output: Tensor<Dtype = f32, Tape = T::Tape>,
{
    type Output = F::Output;

    /// Calls forward on `F` and `R` and then sums their result: `F(x) + R(x)`
    fn forward(&self, x: T) -> Self::Output {
        add(self.f.forward(x.with_empty_tape()), self.r.forward(x))
    }
}

impl<F, R, T> ModuleMut<T> for GeneralizedResidual<F, R>
where
    T: Tensor<Dtype = f32>,
    F: ModuleMut<T>,
    R: ModuleMut<T, Output = F::Output>,
    F::Output: Tensor<Dtype = f32, Tape = T::Tape>,
{
    type Output = F::Output;

    fn forward_mut(&mut self, x: T) -> Self::Output {
        add(
            self.f.forward_mut(x.with_empty_tape()),
            self.r.forward_mut(x),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::{prelude::StdRng, SeedableRng};

    #[test]
    fn test_reset_generalized_residual() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut model: GeneralizedResidual<Linear<2, 5>, Linear<2, 5>> = Default::default();
        assert_eq!(model.f.weight.data(), &[[0.0; 2]; 5]);
        assert_eq!(model.f.bias.data(), &[0.0; 5]);
        assert_eq!(model.r.weight.data(), &[[0.0; 2]; 5]);
        assert_eq!(model.r.bias.data(), &[0.0; 5]);

        model.reset_params(&mut rng);
        assert_ne!(model.f.weight.data(), &[[0.0; 2]; 5]);
        assert_ne!(model.f.bias.data(), &[0.0; 5]);
        assert_ne!(model.r.weight.data(), &[[0.0; 2]; 5]);
        assert_ne!(model.r.bias.data(), &[0.0; 5]);
    }

    #[test]
    fn test_generalized_residual_gradients() {
        let mut rng = StdRng::seed_from_u64(0);

        let mut model: GeneralizedResidual<Linear<2, 2>, Linear<2, 2>> = Default::default();
        model.reset_params(&mut rng);

        let x: Tensor2D<4, 2> = TensorCreator::randn(&mut rng);
        let y = model.forward_mut(x.trace());

        #[rustfmt::skip]
        assert_close(y.data(), &[[-0.81360567, -1.1473482], [1.0925694, 0.17383915], [-0.32519114, 0.49806428], [0.08259219, -0.7277866]]);

        let g = backward(y.mean());
        assert_close(g.ref_gradient(&x), &[[0.15889636, 0.062031522]; 4]);
        assert_close(g.ref_gradient(&model.f.weight), &[[-0.025407, 0.155879]; 2]);
        assert_close(g.ref_gradient(&model.f.bias), &[0.5; 2]);
        assert_close(g.ref_gradient(&model.r.weight), &[[-0.025407, 0.155879]; 2]);
        assert_close(g.ref_gradient(&model.r.bias), &[0.5; 2]);
    }
}
