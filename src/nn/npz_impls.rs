use super::{
    modules::*,
    npz::{LoadFromNpz, SaveToNpz},
    *,
};
use crate::{
    shapes::Dtype,
    tensor::{
        numpy::{NpzError, NumpyDtype},
        CopySlice,
    },
};
use std::format;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive, ZipWriter};

impl<T: ZeroSizedModule> SaveToNpz for T {}
impl<T: ZeroSizedModule> LoadFromNpz for T {}

impl<const C: usize, E: Dtype + NumpyDtype, D: CopySlice<E>> SaveToNpz for BatchNorm2D<C, E, D> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut zip::ZipWriter<W>) -> ZipResult<()> {
        self.scale.write_to_npz(w, format!("{p}scale.npy"))?;
        self.bias.write_to_npz(w, format!("{p}bias.npy"))?;
        self.running_mean
            .write_to_npz(w, format!("{p}running_mean.npy"))?;
        self.running_var
            .write_to_npz(w, format!("{p}running_var.npy"))?;
        Ok(())
    }
}

impl<const C: usize, E: Dtype + NumpyDtype, D: CopySlice<E>> LoadFromNpz for BatchNorm2D<C, E, D> {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.scale.read_from_npz(r, format!("{p}scale.npy"))?;
        self.bias.read_from_npz(r, format!("{p}bias.npy"))?;
        self.running_mean
            .read_from_npz(r, format!("{p}running_mean.npy"))?;
        self.running_var
            .read_from_npz(r, format!("{p}running_var.npy"))?;
        Ok(())
    }
}

#[cfg(feature = "nightly")]
impl<
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        E: Dtype + NumpyDtype,
        D: CopySlice<E>,
    > SaveToNpz for Conv2D<I, O, K, S, P, E, D>
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.weight.write_to_npz(w, format!("{p}weight.npy"))?;
        self.bias.write_to_npz(w, format!("{p}bias.npy"))?;
        Ok(())
    }
}

#[cfg(feature = "nightly")]
impl<
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        E: Dtype + NumpyDtype,
        D: CopySlice<E>,
    > LoadFromNpz for Conv2D<I, O, K, S, P, E, D>
{
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.weight.read_from_npz(r, format!("{p}weight.npy"))?;
        self.bias.read_from_npz(r, format!("{p}bias.npy"))?;
        Ok(())
    }
}

impl<F: SaveToNpz, R: SaveToNpz> SaveToNpz for GeneralizedResidual<F, R> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.f.write(&format!("{p}.f"), w)?;
        self.r.write(&format!("{p}.r"), w)
    }
}

impl<F: LoadFromNpz, R: LoadFromNpz> LoadFromNpz for GeneralizedResidual<F, R> {
    fn read<Z: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<Z>) -> Result<(), NpzError> {
        self.f.read(&format!("{p}.f"), r)?;
        self.r.read(&format!("{p}.r"), r)
    }
}

impl<const M: usize, E: Dtype + NumpyDtype, D: CopySlice<E>> SaveToNpz for LayerNorm1D<M, E, D> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.gamma.write_to_npz(w, format!("{p}gamma.npy"))?;
        self.beta.write_to_npz(w, format!("{p}beta.npy"))?;
        Ok(())
    }
}

impl<const M: usize, E: Dtype + NumpyDtype, D: CopySlice<E>> LoadFromNpz for LayerNorm1D<M, E, D> {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.gamma.read_from_npz(r, format!("{p}gamma.npy"))?;
        self.beta.read_from_npz(r, format!("{p}beta.npy"))?;
        Ok(())
    }
}

impl<const I: usize, const O: usize, E: Dtype + NumpyDtype, D: CopySlice<E>> SaveToNpz
    for Linear<I, O, E, D>
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.weight.write_to_npz(w, format!("{p}weight.npy"))?;
        self.bias.write_to_npz(w, format!("{p}bias.npy"))?;
        Ok(())
    }
}

impl<const I: usize, const O: usize, E: Dtype + NumpyDtype, D: CopySlice<E>> LoadFromNpz
    for Linear<I, O, E, D>
{
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.weight.read_from_npz(r, format!("{p}weight.npy"))?;
        self.bias.read_from_npz(r, format!("{p}bias.npy"))?;
        Ok(())
    }
}

macro_rules! tuple_npz_impl {
    ([$($name:ident),+], [$($idx:tt),+]) => {
impl<$($name: SaveToNpz),+> SaveToNpz for ($($name,)+) {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        $(self.$idx.write(&format!("{p}{}.", $idx), w)?;)+
        Ok(())
    }
}

impl<$($name: LoadFromNpz),+> LoadFromNpz for ($($name,)+) {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        $(self.$idx.read(&format!("{p}{}.", $idx), r)?;)+
        Ok(())
    }
}
    };
}

tuple_npz_impl!([A, B], [0, 1]);
tuple_npz_impl!([A, B, C], [0, 1, 2]);
tuple_npz_impl!([A, B, C, D], [0, 1, 2, 3]);
tuple_npz_impl!([A, B, C, D, E], [0, 1, 2, 3, 4]);
tuple_npz_impl!([A, B, C, D, E, F], [0, 1, 2, 3, 4, 5]);

impl<T: SaveToNpz, const N: usize> SaveToNpz for Repeated<T, N> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        for i in 0..N {
            self.modules[i].write(&format!("{p}{i}."), w)?;
        }
        Ok(())
    }
}

impl<T: LoadFromNpz, const N: usize> LoadFromNpz for Repeated<T, N> {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        for i in 0..N {
            self.modules[i].read(&format!("{p}{i}."), r)?;
        }
        Ok(())
    }
}

impl<F: SaveToNpz> SaveToNpz for Residual<F> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.0.write(&format!("{p}.0"), w)
    }
}

impl<F: LoadFromNpz> LoadFromNpz for Residual<F> {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.0.read(&format!("{p}.0"), r)
    }
}

impl<T: SaveToNpz> SaveToNpz for SplitInto<T> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.0.write(&format!("{p}.0"), w)
    }
}

impl<T: LoadFromNpz> LoadFromNpz for SplitInto<T> {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.0.read(&format!("{p}.0"), r)
    }
}

impl<T: SaveToNpz> SaveToNpz for AddInto<T> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.0.write(&format!("{p}.0"), w)
    }
}

impl<T: LoadFromNpz> LoadFromNpz for AddInto<T> {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.0.read(&format!("{p}.0"), r)
    }
}

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const F: usize, const L: usize, E, D: CopySlice<E>> SaveToNpz
    for TransformerDecoder<M, H, F, L, E, D>
where
    E: Dtype + NumpyDtype,
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.0.write(&format!("{p}.0"), w)
    }
}

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const F: usize, E: Dtype + NumpyDtype, D: CopySlice<E>>
    SaveToNpz for TransformerDecoderBlock<M, H, F, E, D>
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.self_attn.write(&format!("{p}self_attn."), w)?;
        self.norm1.write(&format!("{p}norm1."), w)?;
        self.mh_attn.write(&format!("{p}mh_attn."), w)?;
        self.norm2.write(&format!("{p}norm2."), w)?;
        self.ff.0 .0.write(&format!("{p}linear1."), w)?;
        self.ff.0 .2.write(&format!("{p}linear2."), w)?;
        self.norm3.write(&format!("{p}norm3."), w)?;
        Ok(())
    }
}

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const F: usize, E: Dtype + NumpyDtype, D: CopySlice<E>>
    LoadFromNpz for TransformerDecoderBlock<M, H, F, E, D>
{
    fn read<R: Read + Seek>(&mut self, pre: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.self_attn.read(&format!("{pre}self_attn."), r)?;
        self.norm1.read(&format!("{pre}norm1."), r)?;
        self.mh_attn.read(&format!("{pre}mh_attn."), r)?;
        self.norm2.read(&format!("{pre}norm2."), r)?;
        self.ff.0 .0.read(&format!("{pre}linear1."), r)?;
        self.ff.0 .2.read(&format!("{pre}linear2."), r)?;
        self.norm3.read(&format!("{pre}norm3."), r)?;
        Ok(())
    }
}

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const F: usize, const L: usize, E, D: CopySlice<E>> LoadFromNpz
    for TransformerDecoder<M, H, F, L, E, D>
where
    E: Dtype + NumpyDtype,
{
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.0.read(&format!("{p}.0"), r)
    }
}

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const F: usize, E: Dtype + NumpyDtype, D: CopySlice<E>>
    SaveToNpz for TransformerEncoderBlock<M, H, F, E, D>
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.self_attn.write(&format!("{p}self_attn."), w)?;
        self.norm1.write(&format!("{p}norm1."), w)?;
        self.norm2.write(&format!("{p}norm2."), w)?;
        self.ff.0 .0.write(&format!("{p}linear1."), w)?;
        self.ff.0 .2.write(&format!("{p}linear2."), w)?;
        Ok(())
    }
}

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const F: usize, E: Dtype + NumpyDtype, D: CopySlice<E>>
    LoadFromNpz for TransformerEncoderBlock<M, H, F, E, D>
{
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.self_attn.read(&format!("{p}self_attn."), r)?;
        self.norm1.read(&format!("{p}norm1."), r)?;
        self.norm2.read(&format!("{p}norm2."), r)?;
        self.ff.0 .0.read(&format!("{p}linear1."), r)?;
        self.ff.0 .2.read(&format!("{p}linear2."), r)?;
        Ok(())
    }
}

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const K: usize, const V: usize, E, D: CopySlice<E>> SaveToNpz
    for MultiHeadAttention<M, H, K, V, E, D>
where
    E: Dtype + NumpyDtype,
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.w_q.write(&format!("{p}w_q."), w)?;
        self.w_k.write(&format!("{p}w_k."), w)?;
        self.w_v.write(&format!("{p}w_v."), w)?;
        self.w_o.write(&format!("{p}w_o."), w)?;
        Ok(())
    }
}

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const K: usize, const V: usize, E, D: CopySlice<E>> LoadFromNpz
    for MultiHeadAttention<M, H, K, V, E, D>
where
    E: Dtype + NumpyDtype,
{
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.w_q.read(&format!("{p}w_q."), r)?;
        self.w_k.read(&format!("{p}w_k."), r)?;
        self.w_v.read(&format!("{p}w_v."), r)?;
        self.w_o.read(&format!("{p}w_o."), r)?;
        Ok(())
    }
}

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const A: usize, const B: usize, const F: usize, E, D> SaveToNpz
    for Transformer<M, H, A, B, F, E, D>
where
    E: Dtype + NumpyDtype,
    D: CopySlice<E>,
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.encoder.write(&format!("{p}encoder."), w)?;
        self.decoder.write(&format!("{p}decoder."), w)?;
        Ok(())
    }
}

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const A: usize, const B: usize, const F: usize, E, D>
    LoadFromNpz for Transformer<M, H, A, B, F, E, D>
where
    E: Dtype + NumpyDtype,
    D: CopySlice<E>,
{
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.encoder.read(&format!("{p}encoder."), r)?;
        self.decoder.read(&format!("{p}decoder."), r)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        nn::{builders::*, *},
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
        M::Built: Module<Tensor<S, E, D>> + SaveToNpz + LoadFromNpz,
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
