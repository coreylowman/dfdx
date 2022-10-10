use crate::gradients::{CanUpdateWithGradients, GradientProvider, UnusedTensors};
use crate::prelude::*;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive, ZipWriter};

/// Repeats `T` `N` times. This requires that `T`'s input is the same as it's output.
///
/// # Generics
/// - `T` the [Module] to repeat
/// - `N` the number of times to repeat `T`.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// type Model = Repeated<(Linear<10, 10>, ReLU), 5>;
/// let model: Model = Default::default();
/// let out: Tensor1D<10> = model.forward(Tensor1D::zeros());
/// ```
#[derive(Debug, Clone)]
pub struct Repeated<T, const N: usize> {
    pub modules: Vec<T>,
}

impl<T: Default, const N: usize> Default for Repeated<T, N> {
    fn default() -> Self {
        let mut modules = Vec::with_capacity(N);
        for _ in 0..N {
            modules.push(Default::default());
        }
        Self { modules }
    }
}

impl<T, const N: usize> std::ops::Index<usize> for Repeated<T, N> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.modules[index]
    }
}

impl<T: ResetParams, const N: usize> ResetParams for Repeated<T, N> {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        for i in 0..N {
            self.modules[i].reset_params(rng);
        }
    }
}

impl<T: CanUpdateWithGradients, const N: usize> CanUpdateWithGradients for Repeated<T, N> {
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        for i in 0..N {
            self.modules[i].update(grads, unused);
        }
    }
}

impl<T: SaveToNpz, const N: usize> SaveToNpz for Repeated<T, N> {
    /// Calls `SaveToNpz::write(self.modules[i], ...)` on each sub module. See [SaveToNpz].
    ///
    /// E.g. for a two items with `base == ""`, this will call:
    /// 1. `self.modules[0].write("0.", w)`
    /// 2. `self.modules[1].write("1.", w)`
    fn write<W: Write + Seek>(&self, base: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        for i in 0..N {
            self.modules[i].write(&format!("{base}{i}."), w)?;
        }
        Ok(())
    }
}

impl<T: LoadFromNpz, const N: usize> LoadFromNpz for Repeated<T, N> {
    /// Calls `LoadFromNpz::read(self.modules[i], ...)` on each sub module. See [LoadFromNpz].
    ///
    /// E.g. for a two items with `base == ""`, this will call:
    /// 1. `self.modules[0].read("0.", r)`
    /// 2. `self.modules[1].read("1.", r)`
    fn read<R>(&mut self, base: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError>
    where
        R: Read + Seek,
    {
        for i in 0..N {
            self.modules[i].read(&format!("{base}{i}."), r)?;
        }
        Ok(())
    }
}

impl<Input, T: Module<Input, Output = Input>, const N: usize> Module<Input> for Repeated<T, N> {
    type Output = T::Output;
    fn forward(&self, mut x: Input) -> Self::Output {
        for i in 0..N {
            x = self.modules[i].forward(x);
        }
        x
    }
}

impl<Input, T: ModuleMut<Input, Output = Input>, const N: usize> ModuleMut<Input>
    for Repeated<T, N>
{
    type Output = T::Output;
    fn forward_mut(&mut self, mut x: Input) -> Self::Output {
        for i in 0..N {
            x = self.modules[i].forward_mut(x);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::tests::SimpleGradients;
    use crate::unique_id::HasUniqueId;
    use rand::{prelude::StdRng, SeedableRng};
    use std::fs::File;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_and_reset() {
        let mut rng = StdRng::seed_from_u64(0);

        type Model = Repeated<(Linear<3, 3>, ReLU), 5>;
        let mut m: Model = Default::default();

        for i in 0..5 {
            assert_eq!(m.modules[i].0.weight.data(), &[[0.0; 3]; 3]);
            assert_eq!(m.modules[i].0.bias.data(), &[0.0; 3]);
        }

        m.reset_params(&mut rng);

        for i in 0..5 {
            assert_ne!(m.modules[i].0.weight.data(), &[[0.0; 3]; 3]);
            assert_ne!(m.modules[i].0.bias.data(), &[0.0; 3]);
        }
    }

    #[test]
    fn test_forward() {
        type Model = Repeated<(Linear<3, 3>, ReLU), 5>;

        let mut rng = StdRng::seed_from_u64(0);
        let mut m: Model = Default::default();
        m.reset_params(&mut rng);

        let x = Tensor1D::zeros();
        let x = m.modules[0].forward(x);
        let x = m.modules[1].forward(x);
        let x = m.modules[2].forward(x);
        let x = m.modules[3].forward(x);
        let x = m.modules[4].forward(x);

        assert_eq!(x.data(), m.forward_mut(Tensor1D::zeros()).data());
    }

    #[test]
    fn test_save_repeated() {
        let model: Repeated<Linear<3, 3>, 4> = Default::default();
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
            &[
                "0.bias.npy",
                "0.weight.npy",
                "1.bias.npy",
                "1.weight.npy",
                "2.bias.npy",
                "2.weight.npy",
                "3.bias.npy",
                "3.weight.npy",
            ]
        );
    }

    #[test]
    fn test_load_repeated() {
        type Model = Repeated<Linear<3, 3>, 4>;

        let mut rng = StdRng::seed_from_u64(0);
        let mut saved_model: Model = Default::default();
        saved_model.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model.save(file.path().to_str().unwrap()).is_ok());

        let mut loaded_model: Model = Default::default();
        assert!(loaded_model.load(file.path().to_str().unwrap()).is_ok());
        for i in 0..4 {
            assert_eq!(
                loaded_model.modules[i].weight.data(),
                saved_model.modules[i].weight.data()
            );
            assert_eq!(
                loaded_model.modules[i].bias.data(),
                saved_model.modules[i].bias.data()
            );
        }
    }

    #[test]
    fn test_repeated_missing_gradients() {
        let mut model: Repeated<Linear<5, 5>, 3> = Default::default();
        let mut g: SimpleGradients = Default::default();

        // no gradients present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused);
        assert_eq!(
            &unused.ids,
            &[
                *model[0].weight.id(),
                *model[0].bias.id(),
                *model[1].weight.id(),
                *model[1].bias.id(),
                *model[2].weight.id(),
                *model[2].bias.id(),
            ]
        );

        // weight gradient is present
        for i in 0..3 {
            g.0.mut_gradient(&model[i].weight);
        }

        let mut unused = Default::default();
        model.update(&mut g, &mut unused);
        assert_eq!(
            &unused.ids,
            &[
                *model[0].bias.id(),
                *model[1].bias.id(),
                *model[2].bias.id()
            ]
        );

        // all gradients present
        for i in 0..3 {
            g.0.mut_gradient(&model[i].weight);
            g.0.mut_gradient(&model[i].bias);
        }

        let mut unused = Default::default();
        model.update(&mut g, &mut unused);
        assert!(unused.is_empty());
    }
}
