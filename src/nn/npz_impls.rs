use super::{
    npz::{LoadFromNpz, SaveToNpz},
    *,
};
use crate::{tensor::numpy::NpzError, tensor_ops::Device};
use std::format;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive, ZipWriter};

impl<T: ZeroSizedModule> SaveToNpz for T {}
impl<T: ZeroSizedModule> LoadFromNpz for T {}

impl<const C: usize, D: Device<f32>> SaveToNpz for BatchNorm2D<C, D> {
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

impl<const C: usize, D: Device<f32>> LoadFromNpz for BatchNorm2D<C, D> {
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
        D: Device<f32>,
    > SaveToNpz for Conv2D<I, O, K, S, P, D>
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
        D: Device<f32>,
    > LoadFromNpz for Conv2D<I, O, K, S, P, D>
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

impl<const M: usize, D: Device<f32>> SaveToNpz for LayerNorm1D<M, D> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.gamma.write_to_npz(w, format!("{p}gamma.npy"))?;
        self.beta.write_to_npz(w, format!("{p}beta.npy"))?;
        Ok(())
    }
}

impl<const M: usize, D: Device<f32>> LoadFromNpz for LayerNorm1D<M, D> {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.gamma.read_from_npz(r, format!("{p}gamma.npy"))?;
        self.beta.read_from_npz(r, format!("{p}beta.npy"))?;
        Ok(())
    }
}

impl<const I: usize, const O: usize, D: Device<f32>> SaveToNpz for Linear<I, O, D> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.weight.write_to_npz(w, format!("{p}weight.npy"))?;
        self.bias.write_to_npz(w, format!("{p}bias.npy"))?;
        Ok(())
    }
}

impl<const I: usize, const O: usize, D: Device<f32>> LoadFromNpz for Linear<I, O, D> {
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
impl<const M: usize, const H: usize, const F: usize, const L: usize, D: Device<f32>> SaveToNpz
    for TransformerDecoder<M, H, F, L, D>
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.0.write(&format!("{p}.0"), w)
    }
}

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const F: usize, D: Device<f32>> SaveToNpz
    for TransformerDecoderBlock<M, H, F, D>
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
impl<const M: usize, const H: usize, const F: usize, D: Device<f32>> LoadFromNpz
    for TransformerDecoderBlock<M, H, F, D>
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
impl<const M: usize, const H: usize, const F: usize, const L: usize, D: Device<f32>> LoadFromNpz
    for TransformerDecoder<M, H, F, L, D>
{
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.0.read(&format!("{p}.0"), r)
    }
}

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const F: usize, D: Device<f32>> SaveToNpz
    for TransformerEncoderBlock<M, H, F, D>
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
impl<const M: usize, const H: usize, const F: usize, D: Device<f32>> LoadFromNpz
    for TransformerEncoderBlock<M, H, F, D>
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
impl<const M: usize, const H: usize, const K: usize, const V: usize, D: Device<f32>> SaveToNpz
    for MultiHeadAttention<M, H, K, V, D>
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
impl<const M: usize, const H: usize, const K: usize, const V: usize, D: Device<f32>> LoadFromNpz
    for MultiHeadAttention<M, H, K, V, D>
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
impl<
        const M: usize,
        const H: usize,
        const E: usize,
        const D: usize,
        const F: usize,
        Dev: Device<f32>,
    > SaveToNpz for Transformer<M, H, E, D, F, Dev>
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.encoder.write(&format!("{p}encoder."), w)?;
        self.decoder.write(&format!("{p}decoder."), w)?;
        Ok(())
    }
}

#[cfg(feature = "nightly")]
impl<
        const M: usize,
        const H: usize,
        const E: usize,
        const D: usize,
        const F: usize,
        Dev: Device<f32>,
    > LoadFromNpz for Transformer<M, H, E, D, F, Dev>
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
        prelude::{AsArray, RandnTensor},
        shapes::{Dtype, Rank1, Rank3, Shape},
        tensor::Tensor,
        tensor_ops::Device,
        tests::{build_test_device, TestDevice},
    };

    use super::*;
    use tempfile::NamedTempFile;

    fn test_save_load<
        S: Shape + Default,
        E: Dtype,
        D: Device<E>,
        M: ResetParams<D, E> + Module<Tensor<S, E, D>> + SaveToNpz + LoadFromNpz,
    >(
        dev: &D,
    ) where
        M::Output: AsArray,
        <M::Output as AsArray>::Array: PartialEq,
    {
        let x = dev.randn();
        let file = NamedTempFile::new().expect("failed to create tempfile");

        let saved: M = dev.build_module();
        let mut loaded: M = dev.build_module();

        let y = saved.forward(x.clone());

        assert_ne!(loaded.forward(x.clone()).array(), y.array());

        saved.save(file.path()).expect("");
        loaded.load(file.path()).expect("");

        assert_eq!(loaded.forward(x).array(), y.array());
    }

    #[test]
    fn test_batchnorm2d_save_load() {
        let dev = build_test_device!();

        let x = dev.randn::<Rank3<3, 4, 5>>();
        let file = NamedTempFile::new().expect("failed to create tempfile");

        let mut saved: BatchNorm2D<3> = dev.build_module();
        let mut loaded: BatchNorm2D<3> = dev.build_module();

        saved.running_mean.fill_with_uniform(0.0, 1.0);
        saved.running_var.fill_with_uniform(0.0, 1.0);
        saved.scale.fill_with_uniform(0.0, 1.0);
        saved.bias.fill_with_uniform(0.0, 1.0);
        let y = saved.forward(x.clone());

        assert_ne!(loaded.forward(x.clone()).array(), y.array());

        saved.save(file.path()).expect("");
        loaded.load(file.path()).expect("");

        assert_eq!(loaded.forward(x).array(), y.array());
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_save_load_conv() {
        type T = Conv2D<2, 4, 3, 1, 0, TestDevice>;
        let dev = build_test_device!();
        test_save_load::<Rank3<2, 8, 8>, f32, TestDevice, T>(&dev);
    }

    #[test]
    fn test_save_load_generalized_residual() {
        let dev = build_test_device!();
        type T = GeneralizedResidual<Linear<5, 5, TestDevice>, Linear<5, 5, TestDevice>>;
        test_save_load::<Rank1<5>, f32, TestDevice, T>(&dev);
        test_save_load::<Rank1<5>, f32, TestDevice, (T, T)>(&dev);
    }

    #[test]
    fn test_save_load_linear() {
        let dev = build_test_device!();
        type T = Linear<5, 5, TestDevice>;
        test_save_load::<Rank1<5>, f32, TestDevice, T>(&dev);
        test_save_load::<Rank1<5>, f32, TestDevice, (T, T)>(&dev);
    }

    #[test]
    fn test_save_load_tuple() {
        let dev = build_test_device!();
        type T = (
            Linear<1, 2, TestDevice>,
            ReLU,
            Linear<2, 3, TestDevice>,
            (Dropout, Linear<3, 3, TestDevice>, Linear<3, 4, TestDevice>),
        );
        test_save_load::<Rank1<1>, f32, TestDevice, T>(&dev);
    }

    #[test]
    fn test_save_load_layer_norm() {
        type M = LayerNorm1D<3, TestDevice>;
        let dev = build_test_device!();
        let x = dev.randn::<Rank1<3>>();

        let file = NamedTempFile::new().expect("failed to create tempfile");

        let mut saved: M = dev.build_module();
        let mut loaded: M = dev.build_module();

        saved.gamma.fill_with_uniform(0.0, 1.0);
        saved.beta.fill_with_uniform(0.0, 1.0);
        let y = saved.forward(x.clone());

        assert_ne!(loaded.forward(x.clone()).array(), y.array());

        saved.save(file.path()).expect("");
        loaded.load(file.path()).expect("");

        assert_eq!(loaded.forward(x).array(), y.array());
    }

    #[test]
    fn test_save_load_repeated() {
        type T = Repeated<Linear<3, 3, TestDevice>, 4>;
        let dev = build_test_device!();
        test_save_load::<Rank1<3>, f32, TestDevice, T>(&dev);
        test_save_load::<Rank1<3>, f32, TestDevice, (T, T)>(&dev);
    }

    #[test]
    fn test_save_load_residual() {
        type T = Residual<Linear<5, 5, TestDevice>>;
        let dev = build_test_device!();
        test_save_load::<Rank1<5>, f32, TestDevice, T>(&dev);
        test_save_load::<Rank1<5>, f32, TestDevice, (T, T)>(&dev);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_save_load_mha() {
        let dev = build_test_device!();

        let saved: MultiHeadAttention<12, 4, 12, 12, TestDevice> = dev.build_module();

        let file = NamedTempFile::new().expect("failed to create tempfile");
        saved.save(file.path()).expect("");

        let mut loaded: MultiHeadAttention<12, 4, 12, 12, TestDevice> = dev.build_module();

        let q = dev.randn::<Rank3<2, 3, 12>>();
        let k = dev.randn::<Rank3<2, 4, 12>>();
        let v = dev.randn::<Rank3<2, 4, 12>>();
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
        let dev = build_test_device!();

        let mut saved: Transformer<16, 4, 3, 4, 8, TestDevice> = dev.build_module();

        let file = NamedTempFile::new().expect("failed to create tempfile");
        saved.save(file.path()).expect("");

        let mut loaded: Transformer<16, 4, 3, 4, 8, TestDevice> = dev.build_module();

        let src = dev.randn::<Rank3<4, 12, 16>>();
        let tgt = dev.randn::<Rank3<4, 6, 16>>();
        let y1 = saved.forward_mut((src.clone(), tgt.clone()));

        let y2 = loaded.forward_mut((src.clone(), tgt.clone()));
        assert_ne!(y1.array(), y2.array());

        loaded.load(file.path()).expect("");

        let y2 = loaded.forward_mut((src.clone(), tgt.clone()));
        assert_eq!(y1.array(), y2.array());
    }
}
