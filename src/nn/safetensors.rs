use crate::{
    prelude::Device,
    shapes::{Dtype, HasShape, Shape},
    tensor::{
        safetensors::{Error, SafeDtype},
        CopySlice, DeviceStorage, Tensor,
    },
};
use memmap2::MmapOptions;
use safetensors::{
    serialize_to_file,
    tensor::{Dtype as SDtype, SafeTensors, TensorView},
    SafeTensorError,
};
use std::collections::BTreeMap;

use super::tensor_collection::*;

use std::{path::Path, string::String};

struct TensorData {
    dtype: SDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

pub struct Writer {
    tensors: BTreeMap<String, TensorData>,
}

impl Writer {
    pub fn new() -> Self {
        let tensors = BTreeMap::new();
        Self { tensors }
    }

    pub fn add<S: Shape, E: Dtype + SafeDtype, D: DeviceStorage + CopySlice<E>>(
        &mut self,
        key: String,
        tensor: &Tensor<S, E, D>,
    ) {
        let dtype = E::safe_dtype();
        let shape = tensor.shape().concrete().into();
        let data = tensor.as_vec();
        let data: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tdata = TensorData { dtype, shape, data };
        self.tensors.insert(key, tdata);
    }

    pub fn save(&self, path: &Path) -> Result<(), SafeTensorError> {
        let views: BTreeMap<String, TensorView> = self
            .tensors
            .iter()
            .map(|(k, tensor)| {
                (
                    k.clone(),
                    TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data).unwrap(),
                )
            })
            .collect();
        serialize_to_file(&views, &None, path)
    }
}

impl<E: Dtype + SafeDtype, D: Device<E>> TensorVisitor<E, D> for Writer {
    type Viewer = (ViewTensorRef, ViewTensorName);
    type Err = SafeTensorError;
    type E2 = E;
    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        _: TensorOptions<S, E, D>,
        (t, full_path): (&Tensor<S, E, D>, String),
    ) -> Result<Option<Tensor<S, E, D>>, Self::Err> {
        self.add(full_path, t);
        Ok(None)
    }
}

/// Something that can be saved to a `.safetensors`.
///
/// All [super::Module]s in nn implement SaveToSafetensors.
pub trait SaveToSafetensors<E: Dtype + SafeDtype, D: Device<E>>: TensorCollection<E, D> {
    /// Save this object into the `.safetensors` file determined located at `path`.
    ///
    /// Example:
    /// ```ignore
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let model = dev.build_module::<Linear<15, 5>, f32>();
    /// model.save_safetensors("model.safetensors").unwrap();
    /// ```
    fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<(), SafeTensorError> {
        let mut w = Writer::new();
        Self::iter_tensors(&mut RecursiveWalker {
            m: (self, String::new()),
            f: &mut w,
        })?;
        w.save(path.as_ref())?;
        Ok(())
    }
}
impl<E: Dtype + SafeDtype, D: Device<E>, T: TensorCollection<E, D>> SaveToSafetensors<E, D> for T {}

/// Something that can be loaded from a `.safetensors` file.
///
/// All [super::Module]s in nn implement LoadFromSafetensors.
pub trait LoadFromSafetensors<E: Dtype + SafeDtype, D: Device<E>>: TensorCollection<E, D> {
    /// Loads data from a `.safetensors` at the specified `path`.
    ///
    /// Example:
    /// ```ignore
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let mut model = dev.build_module::<Linear<15, 5>, f32>();
    /// model.load_safetensors("model.safetensors").unwrap();
    /// ```
    fn load_safetensors<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Error> {
        let f = std::fs::File::open(path)?;
        let buffer = unsafe { MmapOptions::new().map(&f)? };
        let mut tensors = SafeTensors::deserialize(&buffer)?;

        Self::iter_tensors(&mut RecursiveWalker {
            m: (self, String::new()),
            f: &mut tensors,
        })?;
        Ok(())
    }
}

impl<E: Dtype + SafeDtype, D: Device<E>, T: TensorCollection<E, D>> LoadFromSafetensors<E, D>
    for T
{
}

impl<'data, E: Dtype + SafeDtype, D: Device<E>> TensorVisitor<E, D> for SafeTensors<'data> {
    type Viewer = (ViewTensorMut, ViewTensorName);
    type Err = Error;
    type E2 = E;
    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        _: TensorOptions<S, E, D>,
        (t, full_path): (&mut Tensor<S, E, D>, String),
    ) -> Result<Option<Tensor<S, E, D>>, Self::Err> {
        t.load_safetensor(self, &full_path)?;
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        nn::builders::*,
        shapes::*,
        tensor::{safetensors::SafeDtype, AsArray, SampleTensor, Tensor},
        tensor_ops::Device,
        tests::{TestDevice, TestDtype},
    };
    use rand_distr::{Distribution, Standard, StandardNormal};
    use tempfile::NamedTempFile;

    fn test_save_load<S: ConstShape, E: Dtype + SafeDtype, D: Device<E>, M: BuildOnDevice<D, E>>(
        dev: &D,
    ) where
        M::Built: Module<Tensor<S, E, D>> + SaveToSafetensors<E, D> + LoadFromSafetensors<E, D>,
        <M::Built as Module<Tensor<S, E, D>>>::Output: AsArray,
        StandardNormal: Distribution<E>,
    {
        let x = dev.sample_normal();
        let file = NamedTempFile::new().expect("failed to create tempfile");

        let saved: M::Built = M::build_on_device(dev);
        let mut loaded: M::Built = M::build_on_device(dev);

        let y = saved.forward(x.clone());

        assert_ne!(loaded.forward(x.clone()).array(), y.array());

        saved.save_safetensors(file.path()).expect("");
        loaded.load_safetensors(file.path()).expect("");

        assert_eq!(loaded.forward(x).array(), y.array());
    }

    #[test]
    fn test_batchnorm2d_save_load_safetensors() {
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

        saved.save_safetensors(file.path()).expect("");
        loaded.load_safetensors(file.path()).expect("");

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

        saved.save_safetensors(file.path()).expect("");
        loaded.load_safetensors(file.path()).expect("");

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
        saved.save_safetensors(file.path()).expect("");

        let mut loaded = Model::build_on_device(&dev);

        let q: Tensor<Rank3<2, 3, 12>, TestDtype, _> = dev.sample_normal();
        let k: Tensor<Rank3<2, 4, 12>, TestDtype, _> = dev.sample_normal();
        let v: Tensor<Rank3<2, 4, 12>, TestDtype, _> = dev.sample_normal();
        let y1 = saved.forward((q.clone(), k.clone(), v.clone()));

        let y2 = loaded.forward((q.clone(), k.clone(), v.clone()));
        assert_ne!(y1.array(), y2.array());

        loaded.load_safetensors(file.path()).expect("");

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
        saved.save_safetensors(file.path()).expect("");

        let mut loaded = Model::build_on_device(&dev);

        let src: Tensor<Rank3<4, 12, 16>, TestDtype, _> = dev.sample_normal();
        let tgt: Tensor<Rank3<4, 6, 16>, TestDtype, _> = dev.sample_normal();
        let y1 = saved.forward_mut((src.clone(), tgt.clone()));

        let y2 = loaded.forward_mut((src.clone(), tgt.clone()));
        assert_ne!(y1.array(), y2.array());

        loaded.load_safetensors(file.path()).expect("");

        let y2 = loaded.forward_mut((src.clone(), tgt.clone()));
        assert_eq!(y1.array(), y2.array());
    }
}
