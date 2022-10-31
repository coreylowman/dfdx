use dfdx_macros::CanUpdateWithGradients;
use crate::gradients::{CanUpdateWithGradients, GradientProvider, UnusedTensors};
use crate::prelude::*;

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
#[derive(Debug, Clone, Default, CanUpdateWithGradients)]
pub struct Residual<F>(pub F);


impl<F: ResetParams> ResetParams for Residual<F> {
    /// Pass through to `F`'s [ResetParams].
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.0.reset_params(rng);
    }
}

impl<T: Tensor<Dtype = f32>, F: Module<T, Output = T>> Module<T> for Residual<F> {
    type Output = F::Output;
    fn forward(&self, x: T) -> Self::Output {
        add(self.0.forward(x.with_empty_tape()), x)
    }
}

impl<T: Tensor<Dtype = f32>, F: ModuleMut<T, Output = T>> ModuleMut<T> for Residual<F> {
    type Output = F::Output;
    fn forward_mut(&mut self, x: T) -> Self::Output {
        add(self.0.forward_mut(x.with_empty_tape()), x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::{prelude::StdRng, SeedableRng};

    #[test]
    fn test_residual_reset() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut model: Residual<Linear<2, 5>> = Default::default();
        assert_eq!(model.0.weight.data(), &[[0.0; 2]; 5]);
        assert_eq!(model.0.bias.data(), &[0.0; 5]);

        model.reset_params(&mut rng);
        assert_ne!(model.0.weight.data(), &[[0.0; 2]; 5]);
        assert_ne!(model.0.bias.data(), &[0.0; 5]);
    }

    #[test]
    fn test_residual_gradients() {
        let mut rng = StdRng::seed_from_u64(0);

        let mut model: Residual<Linear<2, 2>> = Default::default();
        model.reset_params(&mut rng);

        let x: Tensor2D<4, 2> = TensorCreator::randn(&mut rng);
        let y = model.forward_mut(x.trace());

        #[rustfmt::skip]
        assert_close(y.data(), &[[0.25372928, -2.4258814],[1.7892148, -2.6242268],[1.5131638, 0.23407778],[3.4201493, 1.597525]]);

        let g = backward(y.mean());
        assert_close(g.ref_gradient(&model.0.weight), &[[0.475242, -0.075136]; 2]);
        assert_close(g.ref_gradient(&model.0.bias), &[0.5; 2]);
        assert_close(g.ref_gradient(&x), &[[0.18806472, 0.21419683]; 4]);
    }
}
