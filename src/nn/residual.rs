use crate::gradients::{CanUpdateWithGradients, GradientProvider, UnusedTensors};
use crate::prelude::*;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive, ZipWriter};

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

impl<F: CanUpdateWithGradients> CanUpdateWithGradients for Residual<F> {
    /// Pass through to `F`'s [CanUpdateWithGradients].
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.0.update(grads, unused);
    }
}

impl<F: ResetParams> ResetParams for Residual<F> {
    /// Pass through to `F`'s [ResetParams].
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.0.reset_params(rng);
    }
}

impl<T: Tensor<Dtype = f32>, F: Module<T, Output = T>> Module<T> for Residual<F> {
    type Output = F::Output;
    fn forward(&self, x: T) -> Self::Output {
        let (x, tape) = x.split_tape();
        add(self.0.forward(x.duplicate().put_tape(tape)), &x)
    }
}

impl<T: Tensor<Dtype = f32>, F: ModuleMut<T, Output = T>> ModuleMut<T> for Residual<F> {
    type Output = F::Output;
    fn forward_mut(&mut self, x: T) -> Self::Output {
        let (x, tape) = x.split_tape();
        add(self.0.forward_mut(x.duplicate().put_tape(tape)), &x)
    }
}

impl<F: SaveToNpz> SaveToNpz for Residual<F> {
    /// Pass through to `F`'s [SaveToNpz].
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.0.write(p, w)
    }
}

impl<F: LoadFromNpz> LoadFromNpz for Residual<F> {
    /// Pass through to `F`'s [LoadFromNpz].
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.0.read(p, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::{prelude::StdRng, SeedableRng};
    use tempfile::NamedTempFile;

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

    #[test]
    fn test_save_load_residual() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut saved_model: Residual<Linear<5, 3>> = Default::default();
        saved_model.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model.save(file.path().to_str().unwrap()).is_ok());

        let mut loaded_model: Residual<Linear<5, 3>> = Default::default();
        assert_ne!(loaded_model.0.weight.data(), saved_model.0.weight.data());
        assert_ne!(loaded_model.0.bias.data(), saved_model.0.bias.data());

        assert!(loaded_model.load(file.path().to_str().unwrap()).is_ok());
        assert_eq!(loaded_model.0.weight.data(), saved_model.0.weight.data());
        assert_eq!(loaded_model.0.bias.data(), saved_model.0.bias.data());
    }
}
