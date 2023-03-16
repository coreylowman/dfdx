use crate::{
    prelude::Device,
    shapes::{Dtype, Shape},
    tensor::{
        numpy::{NpzError, NumpyDtype},
        Tensor,
    },
};

use super::tensor_collection::*;

use std::{
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::Path,
    string::String,
};
use zip::{
    result::{ZipError, ZipResult},
    ZipArchive, ZipWriter,
};

/// Something that can be saved to a `.npz` (which is a `.zip`).
///
/// All [super::Module]s in nn implement SaveToNpz, and the zips are formatted in a `.npz` fashion.
pub trait SaveToNpz<E: Dtype + NumpyDtype, D: Device<E>>: TensorCollection<E, D> {
    /// Save this object into the `.npz` file determined located at `path`.
    ///
    /// Example:
    /// ```ignore
    /// # use dfdx::prelude::*;
    /// let model: (Linear<5, 10>, Linear<10, 5>) = Default::default();
    /// model.save("tst.npz")?;
    /// ```
    fn save<P: AsRef<Path>>(&self, path: P) -> ZipResult<()> {
        let f = std::fs::File::create(path)?;
        let f = BufWriter::new(f);
        let mut zip = ZipWriter::new(f);
        self.write(&mut zip)?;
        zip.finish()?;
        Ok(())
    }

    /// Write this object into [ZipWriter] `w` with a base filename of `filename_prefix`.
    ///
    /// Example:
    /// ```ignore
    /// # use dfdx::prelude::*;
    /// let model: Linear<5, 10> = Default::default();
    /// let mut zip = ZipWriter::new(...);
    /// model.write("0", &mut zip)?;
    /// model.write("1", &mut zip)?;
    /// ```
    /// Will save a zip file with the following files in it:
    /// - `0.weight.npy`
    /// - `0.bias.npy`
    /// - `1.weight.npy`
    /// - `1.bias.npy`
    fn write<W>(&self, w: &mut ZipWriter<W>) -> ZipResult<()>
    where
        W: Write + Seek,
    {
        Self::iter_tensors(&mut RecursiveWalker {
            m: (self, String::new()),
            f: w,
        })?;
        Ok(())
    }
}
impl<E: Dtype + NumpyDtype, D: Device<E>, T: TensorCollection<E, D>> SaveToNpz<E, D> for T {}

/// Something that can be loaded from a `.npz` file (which is a `zip` file).
///
/// All [super::Module]s in nn implement LoadFromNpz, and the zips are formatted in a `.npz` fashion.
pub trait LoadFromNpz<E: Dtype + NumpyDtype, D: Device<E>>: TensorCollection<E, D> {
    /// Loads data from a `.npz` zip archive at the specified `path`.
    ///
    /// Example:
    /// ```ignore
    /// # use dfdx::prelude::*;
    /// let mut model: (Linear<5, 10>, Linear<10, 5>) = Default::default();
    /// model.load("tst.npz")?;
    /// ```
    fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<(), NpzError> {
        let f = std::fs::File::open(path)?;
        let f = BufReader::new(f);
        let mut zip = ZipArchive::new(f)?;
        self.read(&mut zip)?;
        Ok(())
    }

    /// Reads this object from a [ZipArchive]. `r` with a base filename of `filename_prefix`.
    ///
    /// Example:
    /// ```ignore
    /// # use dfdx::prelude::*;
    /// let mut model: Linear<5, 10> = Default::default();
    /// let mut zip = ZipArchive::new(...);
    /// model.read("0", &mut zip)?;
    /// ```
    /// Will try to read data from the following files:
    /// - `0.weight.npy`
    /// - `0.bias.npy`
    fn read<R>(&mut self, r: &mut ZipArchive<R>) -> Result<(), NpzError>
    where
        R: Read + Seek,
    {
        Self::iter_tensors(&mut RecursiveWalker {
            m: (self, String::new()),
            f: r,
        })?;
        Ok(())
    }
}
impl<E: Dtype + NumpyDtype, D: Device<E>, T: TensorCollection<E, D>> LoadFromNpz<E, D> for T {}

impl<W: Write + Seek, E: Dtype + NumpyDtype, D: Device<E>> TensorVisitor<E, D>
    for zip::ZipWriter<W>
{
    type Viewer = (ViewTensorRef, ViewTensorName);
    type Err = ZipError;
    type E2 = E;
    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        _: TensorOptions<S, E, D>,
        (t, full_path): (&Tensor<S, E, D>, String),
    ) -> Result<Option<Tensor<S, E, D>>, Self::Err> {
        t.write_to_npz(self, full_path)?;
        Ok(None)
    }
}

impl<R: Read + Seek, E: Dtype + NumpyDtype, D: Device<E>> TensorVisitor<E, D>
    for zip::ZipArchive<R>
{
    type Viewer = (ViewTensorMut, ViewTensorName);
    type Err = NpzError;
    type E2 = E;
    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        _: TensorOptions<S, E, D>,
        (t, full_path): (&mut Tensor<S, E, D>, String),
    ) -> Result<Option<Tensor<S, E, D>>, Self::Err> {
        t.read_from_npz(self, full_path)?;
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        nn::builders::*,
        shapes::*,
        tensor::{numpy::NumpyDtype, AsArray, SampleTensor, Tensor},
        tensor_ops::Device,
        tests::{TestDevice, TestDtype},
    };
    use rand_distr::{Distribution, Standard, StandardNormal};
    use tempfile::NamedTempFile;

    fn test_save_load<S: ConstShape, E: Dtype + NumpyDtype, D: Device<E>, M: BuildOnDevice<D, E>>(
        dev: &D,
    ) where
        M::Built: Module<Tensor<S, E, D>> + SaveToNpz<E, D> + LoadFromNpz<E, D>,
        <M::Built as Module<Tensor<S, E, D>>>::Output: AsArray,
        StandardNormal: Distribution<E>,
    {
        let x = dev.sample_normal();
        let file = NamedTempFile::new().expect("failed to create tempfile");

        let saved: M::Built = M::build_on_device(dev);
        let mut loaded: M::Built = M::build_on_device(dev);

        let y = saved.forward(x.clone());

        assert_ne!(loaded.forward(x.clone()).array(), y.array());

        saved.save(file.path()).expect("");
        loaded.load(file.path()).expect("");

        assert_eq!(loaded.forward(x).array(), y.array());
    }

    #[test]
    fn test_batchnorm2d_save_load() {
        let dev: TestDevice = Default::default();
        type Model = BatchNorm2D<3>;

        let x: Tensor<Rank3<3, 4, 5>, TestDtype, _> = dev.sample_normal();
        let file = NamedTempFile::new().expect("failed to create tempfile");

        let mut saved = Model::build_on_device(&dev);
        let mut loaded = Model::build_on_device(&dev);

        saved.running_mean.fill_with_distr(Standard);
        saved.running_var.fill_with_distr(Standard);
        saved.scale.fill_with_distr(Standard);
        saved.bias.fill_with_distr(Standard);
        let y = saved.forward(x.clone());

        assert_ne!(loaded.forward(x.clone()).array(), y.array());

        saved.save(file.path()).expect("");
        loaded.load(file.path()).expect("");

        assert_eq!(loaded.forward(x).array(), y.array());
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_save_load_conv() {
        type T = Conv2D<2, 4, 3>;
        let dev: TestDevice = Default::default();
        test_save_load::<Rank3<2, 8, 8>, TestDtype, TestDevice, T>(&dev);
    }

    #[test]
    fn test_save_load_generalized_residual() {
        let dev: TestDevice = Default::default();
        type T = GeneralizedResidual<Linear<5, 5>, Linear<5, 5>>;
        test_save_load::<Rank1<5>, TestDtype, TestDevice, T>(&dev);
        test_save_load::<Rank1<5>, TestDtype, TestDevice, (T, T)>(&dev);
    }

    #[test]
    fn test_save_load_linear() {
        let dev: TestDevice = Default::default();
        type T = Linear<5, 5>;
        test_save_load::<Rank1<5>, TestDtype, TestDevice, T>(&dev);
        test_save_load::<Rank1<5>, TestDtype, TestDevice, (T, T)>(&dev);
    }

    #[test]
    fn test_save_load_tuple() {
        let dev: TestDevice = Default::default();
        type T = (
            (Linear<1, 2>, ReLU, Linear<2, 3>),
            (Dropout, Linear<3, 3>, Linear<3, 4>),
        );
        test_save_load::<Rank1<1>, TestDtype, TestDevice, T>(&dev);
    }

    #[test]
    fn test_save_load_layer_norm() {
        type M = LayerNorm1D<3>;
        let dev: TestDevice = Default::default();
        let x: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();

        let file = NamedTempFile::new().expect("failed to create tempfile");

        let mut saved = M::build_on_device(&dev);
        let mut loaded = M::build_on_device(&dev);

        saved.gamma.fill_with_distr(Standard);
        saved.beta.fill_with_distr(Standard);
        let y = saved.forward(x.clone());

        assert_ne!(loaded.forward(x.clone()).array(), y.array());

        saved.save(file.path()).expect("");
        loaded.load(file.path()).expect("");

        assert_eq!(loaded.forward(x).array(), y.array());
    }

    #[test]
    fn test_save_load_repeated() {
        type T = Repeated<Linear<3, 3>, 4>;
        let dev: TestDevice = Default::default();
        test_save_load::<Rank1<3>, TestDtype, TestDevice, T>(&dev);
        test_save_load::<Rank1<3>, TestDtype, TestDevice, (T, T)>(&dev);
    }

    #[test]
    fn test_save_load_residual() {
        type T = Residual<Linear<5, 5>>;
        let dev: TestDevice = Default::default();
        test_save_load::<Rank1<5>, TestDtype, TestDevice, T>(&dev);
        test_save_load::<Rank1<5>, TestDtype, TestDevice, (T, T)>(&dev);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_save_load_mha() {
        let dev: TestDevice = Default::default();
        type Model = MultiHeadAttention<12, 4>;

        let saved = Model::build_on_device(&dev);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        saved.save(file.path()).expect("");

        let mut loaded = Model::build_on_device(&dev);

        let q: Tensor<Rank3<2, 3, 12>, TestDtype, _> = dev.sample_normal();
        let k: Tensor<Rank3<2, 4, 12>, TestDtype, _> = dev.sample_normal();
        let v: Tensor<Rank3<2, 4, 12>, TestDtype, _> = dev.sample_normal();
        let y1 = saved.forward((q.clone(), k.clone(), v.clone()));

        let y2 = loaded.forward((q.clone(), k.clone(), v.clone()));
        assert_ne!(y1.array(), y2.array());

        loaded.load(file.path()).expect("");

        let y2 = loaded.forward((q.clone(), k.clone(), v.clone()));
        assert_eq!(y1.array(), y2.array());
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_save_load_transformer() {
        let dev: TestDevice = Default::default();
        type Model = Transformer<16, 4, 3, 4, 8>;

        let mut saved = Model::build_on_device(&dev);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        saved.save(file.path()).expect("");

        let mut loaded = Model::build_on_device(&dev);

        let src: Tensor<Rank3<4, 12, 16>, TestDtype, _> = dev.sample_normal();
        let tgt: Tensor<Rank3<4, 6, 16>, TestDtype, _> = dev.sample_normal();
        let y1 = saved.forward_mut((src.clone(), tgt.clone()));

        let y2 = loaded.forward_mut((src.clone(), tgt.clone()));
        assert_ne!(y1.array(), y2.array());

        loaded.load(file.path()).expect("");

        let y2 = loaded.forward_mut((src.clone(), tgt.clone()));
        assert_eq!(y1.array(), y2.array());
    }
}
