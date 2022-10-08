use crate::arrays::Axis;
use crate::devices::{Cpu, FillElements};
use crate::gradients::{CanUpdateWithGradients, GradientProvider, Tape, UnusedTensors};
use crate::prelude::*;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive};

/// Implements layer normalization as described in [Layer Normalization](https://arxiv.org/abs/1607.06450).
///
/// This calls [normalize()] on the last axis of the input to normalize to 0 mean and unit std dev, and then does an element-wise
/// affine transform using learnable parameters [Self::gamma] and [Self::beta].
///
/// [Self::epsilon] is passed to [normalize()] and added to the variance to ensure big enough numbers. It defaults to `1e-5`.
///
/// # Generics
/// - `M` The size of the affine transform tensors.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// let model: LayerNorm1D<5> = Default::default();
/// let x: Tensor1D<5> = Default::default();
/// let _: Tensor1D<5> = model.forward(x);
/// ```
#[derive(Debug, Clone)]
pub struct LayerNorm1D<const M: usize> {
    pub gamma: Tensor1D<M>,
    pub beta: Tensor1D<M>,
    pub epsilon: f32,
}

impl<const M: usize> Default for LayerNorm1D<M> {
    /// Fills [Self::gamma] with 1s and [Self::beta] with 0s and sets [Self::epsilon] to `1e-5`.
    fn default() -> Self {
        Self {
            gamma: TensorCreator::ones(),
            beta: TensorCreator::zeros(),
            epsilon: 1e-5,
        }
    }
}

impl<const M: usize> ResetParams for LayerNorm1D<M> {
    /// Fills [Self::gamma] with 1s and [Self::beta] with 0s.
    fn reset_params<R: rand::Rng>(&mut self, _: &mut R) {
        Cpu::fill(self.gamma.mut_data(), &mut |v| *v = 1.0);
        Cpu::fill(self.beta.mut_data(), &mut |v| *v = 0.0);
    }
}

impl<const M: usize> CanUpdateWithGradients for LayerNorm1D<M> {
    /// Updates [Self::gamma] and [Self::beta].
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.gamma.update(grads, unused);
        self.beta.update(grads, unused);
    }
}

impl<H: Tape, const M: usize> Module<Tensor1D<M, H>> for LayerNorm1D<M> {
    type Output = Tensor1D<M, H>;

    /// Calls:
    /// 1. [normalize()] with [Self::epsilon]
    /// 2. [mul()] with [Self::gamma]
    /// 3. [add()] with [Self::beta]
    fn forward(&self, x: Tensor1D<M, H>) -> Self::Output {
        let x = x.normalize(self.epsilon);
        let x = mul(x, &self.gamma);
        add(x, &self.beta)
    }
}

impl<H: Tape, const B: usize, const M: usize> Module<Tensor2D<B, M, H>> for LayerNorm1D<M> {
    type Output = Tensor2D<B, M, H>;

    /// Calls:
    /// 1. [normalize()] with [Self::epsilon].
    /// 2. [mul()] with [Self::gamma]
    /// 3. [add()] with [Self::beta]
    fn forward(&self, x: Tensor2D<B, M, H>) -> Self::Output {
        let (x, tape) = x.normalize::<Axis<1>>(self.epsilon).split_tape();
        let g: Tensor2D<B, M, H> = self.gamma.duplicate().put_tape(tape).broadcast();
        let (x, tape) = mul(g, &x).split_tape();
        let b = self.beta.duplicate().put_tape(tape).broadcast();
        add(b, &x)
    }
}

impl<H: Tape, const B: usize, const S: usize, const M: usize> Module<Tensor3D<B, S, M, H>>
    for LayerNorm1D<M>
{
    type Output = Tensor3D<B, S, M, H>;

    /// Calls:
    /// 1. [normalize()] with [Self::epsilon].
    /// 2. [add()] with [Self::gamma]
    /// 3. [add()] with [Self::beta]
    fn forward(&self, x: Tensor3D<B, S, M, H>) -> Self::Output {
        let (x, tape) = x.normalize::<Axis<2>>(self.epsilon).split_tape();
        let g: Tensor3D<B, S, M, H> = self.gamma.duplicate().put_tape(tape).broadcast();
        let (x, tape) = mul(g, &x).split_tape();
        let b = self.beta.duplicate().put_tape(tape).broadcast();
        add(b, &x)
    }
}

impl<T, const M: usize> ModuleMut<T> for LayerNorm1D<M>
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;
    fn forward_mut(&mut self, input: T) -> Self::Output {
        self.forward(input)
    }
}

impl<const M: usize> SaveToNpz for LayerNorm1D<M> {
    /// Saves [Self::gamma] to `{pre}gamma.npy` and [Self::beta] to `{pre}beta.npy`
    /// using [npz_fwrite()].
    fn write<W: Write + Seek>(&self, pre: &str, w: &mut zip::ZipWriter<W>) -> ZipResult<()> {
        npz_fwrite(w, format!("{pre}gamma.npy"), self.gamma.data())?;
        npz_fwrite(w, format!("{pre}beta.npy"), self.beta.data())?;
        Ok(())
    }
}

impl<const M: usize> LoadFromNpz for LayerNorm1D<M> {
    /// Reads [Self::gamma] from `{p}gamma.npy` and [Self::beta] from `{p}beta.npy`
    /// using [npz_fread()].
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        npz_fread(r, format!("{p}gamma.npy"), self.gamma.mut_data())?;
        npz_fread(r, format!("{p}beta.npy"), self.beta.mut_data())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unique_id::HasUniqueId;
    use crate::{nn::tests::SimpleGradients, tests::assert_close};
    use rand::{prelude::StdRng, SeedableRng};
    use rand_distr::Standard;
    use std::fs::File;
    use tempfile::NamedTempFile;

    #[test]
    fn test_layer_norm_reset() {
        let mut m: LayerNorm1D<5> = Default::default();
        assert_eq!(m.gamma.data(), &[1.0; 5]);
        assert_eq!(m.beta.data(), &[0.0; 5]);

        let mut rng = StdRng::seed_from_u64(0);
        m.gamma.randomize(&mut rng, &Standard);
        m.beta.randomize(&mut rng, &Standard);

        assert_ne!(m.gamma.data(), &[1.0; 5]);
        assert_ne!(m.beta.data(), &[0.0; 5]);

        m.reset_params(&mut rng);
        assert_eq!(m.gamma.data(), &[1.0; 5]);
        assert_eq!(m.beta.data(), &[0.0; 5]);
    }

    #[test]
    fn test_layer_norm_1d_forward() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut m: LayerNorm1D<5> = Default::default();
        let x: Tensor1D<5> = TensorCreator::randn(&mut rng);
        let r = m.forward_mut(x.trace());
        assert_close(
            r.data(),
            &[0.873304, 0.9879816, -1.6083492, 0.44028836, -0.6932247],
        );
        let g = backward(r.mean());
        assert_close(
            g.ref_gradient(&m.gamma),
            &[0.1746608, 0.19759633, -0.32166985, 0.088057674, -0.13864495],
        );
        assert_close(g.ref_gradient(&m.beta), &[0.2; 5]);
    }

    #[test]
    fn test_layer_norm_2d_forward() {
        let mut rng = StdRng::seed_from_u64(0);
        let m: LayerNorm1D<5> = Default::default();
        let x: Tensor2D<3, 5> = TensorCreator::randn(&mut rng);
        let r = m.forward(x.trace());
        assert_close(
            r.data(),
            &[
                [0.873304, 0.9879816, -1.6083492, 0.44028836, -0.6932247],
                [0.663322, -1.8449169, 0.05217871, 0.056903206, 1.0725129],
                [1.0343355, -1.5559655, -0.40086073, 1.1405537, -0.21806297],
            ],
        );
        let g = backward(r.mean());
        assert_close(
            g.ref_gradient(&m.gamma),
            &[0.1713974, -0.16086, -0.1304687, 0.109183, 0.0107483],
        );
        assert_close(g.ref_gradient(&m.beta), &[0.2; 5]);
    }

    #[test]
    fn test_save_layer_norm() {
        let model: LayerNorm1D<13> = Default::default();
        let file = NamedTempFile::new().expect("failed to create tempfile");
        model
            .save(file.path().to_str().unwrap())
            .expect("failed to save model");
        let f = File::open(file.path()).expect("failed to open resulting file");
        let zip = ZipArchive::new(f).expect("failed to create zip archive from file");
        let mut names = zip.file_names().collect::<Vec<&str>>();
        names.sort_unstable();
        assert_eq!(&names, &["beta.npy", "gamma.npy",]);
    }

    #[test]
    fn test_save_layer_norm_tuple() {
        let model: (LayerNorm1D<5>, LayerNorm1D<13>) = Default::default();
        let file = NamedTempFile::new().expect("failed to create tempfile");
        model
            .save(file.path().to_str().unwrap())
            .expect("failed to save model");
        let f = File::open(file.path()).expect("failed to open resulting file");
        let zip = ZipArchive::new(f).expect("failed to create zip archive from file");
        let mut names = zip.file_names().collect::<Vec<&str>>();
        names.sort_unstable();
        assert_eq!(
            &names,
            &["0.beta.npy", "0.gamma.npy", "1.beta.npy", "1.gamma.npy"]
        );
    }

    #[test]
    fn test_load_layer_norm() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut saved_model: LayerNorm1D<13> = Default::default();
        saved_model.gamma.randomize(&mut rng, &Standard);
        saved_model.beta.randomize(&mut rng, &Standard);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model.save(file.path().to_str().unwrap()).is_ok());

        let mut loaded_model: LayerNorm1D<13> = Default::default();
        assert!(loaded_model.gamma.data() != saved_model.gamma.data());
        assert!(loaded_model.beta.data() != saved_model.beta.data());

        assert!(loaded_model.load(file.path().to_str().unwrap()).is_ok());
        assert_eq!(loaded_model.gamma.data(), saved_model.gamma.data());
        assert_eq!(loaded_model.beta.data(), saved_model.beta.data());
    }

    #[test]
    fn test_layer_norm_missing_gradients() {
        let mut model: LayerNorm1D<5> = Default::default();
        let mut g: SimpleGradients = Default::default();

        // no gradients present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused);
        assert_eq!(&unused.ids, &[*model.gamma.id(), *model.beta.id()]);

        g.0.mut_gradient(&model.gamma);

        // weight gradient is present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused);
        assert_eq!(&unused.ids, &[*model.beta.id()]);

        g.0.mut_gradient(&model.gamma);
        g.0.mut_gradient(&model.beta);

        // all gradients present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused);
        assert!(unused.is_empty());
    }
}
