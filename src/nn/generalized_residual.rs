use crate::gradients::{CanUpdateWithGradients, GradientProvider, UnusedTensors};
use crate::prelude::*;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive, ZipWriter};

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
        let (x, tape) = x.split_tape();

        // do R(x) on the tape
        let r_x = self.r.forward(x.duplicate().put_tape(tape));
        let (r_x, tape) = r_x.split_tape();

        // do F(x) on the tape
        let f_x = self.f.forward(x.put_tape(tape));

        add(f_x, &r_x)
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
        let (x, tape) = x.split_tape();

        // do R(x) on the tape
        let r_x = self.r.forward_mut(x.duplicate().put_tape(tape));
        let (r_x, tape) = r_x.split_tape();

        // do F(x) on the tape
        let f_x = self.f.forward_mut(x.put_tape(tape));

        add(f_x, &r_x)
    }
}

impl<F: SaveToNpz, R: SaveToNpz> SaveToNpz for GeneralizedResidual<F, R> {
    /// Pass through to `F`/`R`'s [SaveToNpz].
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.f.write(&format!("{p}.f"), w)?;
        self.r.write(&format!("{p}.r"), w)
    }
}

impl<F: LoadFromNpz, R: LoadFromNpz> LoadFromNpz for GeneralizedResidual<F, R> {
    /// Pass through to `F`/`R`'s [LoadFromNpz].
    fn read<Z: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<Z>) -> Result<(), NpzError> {
        self.f.read(&format!("{p}.f"), r)?;
        self.r.read(&format!("{p}.r"), r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::{prelude::StdRng, SeedableRng};
    use tempfile::NamedTempFile;

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

    #[test]
    fn test_save_load_generalized_residual() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut saved_model: GeneralizedResidual<Linear<5, 3>, Linear<5, 3>> = Default::default();
        saved_model.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model.save(file.path().to_str().unwrap()).is_ok());

        let mut loaded_model: GeneralizedResidual<Linear<5, 3>, Linear<5, 3>> = Default::default();
        assert_ne!(loaded_model.f.weight.data(), saved_model.f.weight.data());
        assert_ne!(loaded_model.f.bias.data(), saved_model.f.bias.data());
        assert_ne!(loaded_model.r.weight.data(), saved_model.r.weight.data());
        assert_ne!(loaded_model.r.bias.data(), saved_model.r.bias.data());

        assert!(loaded_model.load(file.path().to_str().unwrap()).is_ok());
        assert_eq!(loaded_model.f.weight.data(), saved_model.f.weight.data());
        assert_eq!(loaded_model.f.bias.data(), saved_model.f.bias.data());
        assert_eq!(loaded_model.r.weight.data(), saved_model.r.weight.data());
        assert_eq!(loaded_model.r.bias.data(), saved_model.r.bias.data());
    }
}
