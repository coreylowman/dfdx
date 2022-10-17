use super::npz::{npz_fread, npz_fwrite, LoadFromNpz, SaveToNpz};
use crate::prelude::*;
use std::format;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive, ZipWriter};

// nightly includes
#[cfg(not(feature = "nightly"))]
use super::conv::Conv2D;
#[cfg(not(feature = "nightly"))]
use super::flatten::*;
#[cfg(not(feature = "nightly"))]
use super::pool2d::*;
#[cfg(not(feature = "nightly"))]
use super::transformer::*;

impl<const C: usize> SaveToNpz for BatchNorm2D<C> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut zip::ZipWriter<W>) -> ZipResult<()> {
        npz_fwrite(w, format!("{p}scale.npy"), self.scale.data())?;
        npz_fwrite(w, format!("{p}bias.npy"), self.bias.data())?;
        npz_fwrite(w, format!("{p}running_mean.npy"), self.running_mean.data())?;
        npz_fwrite(w, format!("{p}running_var.npy"), self.running_var.data())?;
        Ok(())
    }
}

impl<const C: usize> LoadFromNpz for BatchNorm2D<C> {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        npz_fread(r, format!("{p}scale.npy"), self.scale.mut_data())?;
        npz_fread(r, format!("{p}bias.npy"), self.bias.mut_data())?;
        let mean = self.running_mean.mut_data();
        npz_fread(r, format!("{p}running_mean.npy"), mean)?;
        let var = self.running_var.mut_data();
        npz_fread(r, format!("{p}running_var.npy"), var)?;
        Ok(())
    }
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize> SaveToNpz
    for Conv2D<I, O, K, S, P>
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        npz_fwrite(w, format!("{p}weight.npy"), self.weight.data())?;
        npz_fwrite(w, format!("{p}bias.npy"), self.bias.data())?;
        Ok(())
    }
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize> LoadFromNpz
    for Conv2D<I, O, K, S, P>
{
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        npz_fread(r, format!("{p}weight.npy"), self.weight.mut_data())?;
        npz_fread(r, format!("{p}bias.npy"), self.bias.mut_data())?;
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

impl<const M: usize> SaveToNpz for LayerNorm1D<M> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        npz_fwrite(w, format!("{p}gamma.npy"), self.gamma.data())?;
        npz_fwrite(w, format!("{p}beta.npy"), self.beta.data())?;
        Ok(())
    }
}

impl<const M: usize> LoadFromNpz for LayerNorm1D<M> {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        npz_fread(r, format!("{p}gamma.npy"), self.gamma.mut_data())?;
        npz_fread(r, format!("{p}beta.npy"), self.beta.mut_data())?;
        Ok(())
    }
}

impl<const I: usize, const O: usize> SaveToNpz for Linear<I, O> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        npz_fwrite(w, format!("{p}weight.npy"), self.weight.data())?;
        npz_fwrite(w, format!("{p}bias.npy"), self.bias.data())?;
        Ok(())
    }
}

impl<const I: usize, const O: usize> LoadFromNpz for Linear<I, O> {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        npz_fread(r, format!("{p}weight.npy"), self.weight.mut_data())?;
        npz_fread(r, format!("{p}bias.npy"), self.bias.mut_data())?;
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

impl<const M: usize, const H: usize, const F: usize, const L: usize> SaveToNpz
    for TransformerDecoder<M, H, F, L>
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.0.write(&format!("{p}.0"), w)
    }
}

impl<const M: usize, const H: usize, const F: usize> SaveToNpz
    for TransformerDecoderBlock<M, H, F>
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

impl<const M: usize, const H: usize, const F: usize> LoadFromNpz
    for TransformerDecoderBlock<M, H, F>
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

impl<const M: usize, const H: usize, const F: usize, const L: usize> LoadFromNpz
    for TransformerDecoder<M, H, F, L>
{
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.0.read(&format!("{p}.0"), r)
    }
}

impl<const M: usize, const H: usize, const F: usize> SaveToNpz
    for TransformerEncoderBlock<M, H, F>
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

impl<const M: usize, const H: usize, const F: usize> LoadFromNpz
    for TransformerEncoderBlock<M, H, F>
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

impl<const M: usize, const H: usize, const K: usize, const V: usize> SaveToNpz
    for MultiHeadAttention<M, H, K, V>
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.w_q.write(&format!("{p}w_q."), w)?;
        self.w_k.write(&format!("{p}w_k."), w)?;
        self.w_v.write(&format!("{p}w_v."), w)?;
        self.w_o.write(&format!("{p}w_o."), w)?;
        Ok(())
    }
}

impl<const M: usize, const H: usize, const K: usize, const V: usize> LoadFromNpz
    for MultiHeadAttention<M, H, K, V>
{
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.w_q.read(&format!("{p}w_q."), r)?;
        self.w_k.read(&format!("{p}w_k."), r)?;
        self.w_v.read(&format!("{p}w_v."), r)?;
        self.w_o.read(&format!("{p}w_o."), r)?;
        Ok(())
    }
}

impl<const M: usize, const H: usize, const E: usize, const D: usize, const F: usize> SaveToNpz
    for Transformer<M, H, E, D, F>
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.encoder.write(&format!("{p}encoder."), w)?;
        self.decoder.write(&format!("{p}decoder."), w)?;
        Ok(())
    }
}

impl<const M: usize, const H: usize, const E: usize, const D: usize, const F: usize> LoadFromNpz
    for Transformer<M, H, E, D, F>
{
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.encoder.read(&format!("{p}encoder."), r)?;
        self.decoder.read(&format!("{p}decoder."), r)?;
        Ok(())
    }
}

macro_rules! empty_npz_impl {
    ($TyName:ty) => {
        impl SaveToNpz for $TyName {}
        impl LoadFromNpz for $TyName {}
    };
}

empty_npz_impl!(ReLU);
empty_npz_impl!(Sin);
empty_npz_impl!(Cos);
empty_npz_impl!(Ln);
empty_npz_impl!(Exp);
empty_npz_impl!(Sigmoid);
empty_npz_impl!(Tanh);
empty_npz_impl!(Square);
empty_npz_impl!(Sqrt);
empty_npz_impl!(Abs);
empty_npz_impl!(Softmax);
empty_npz_impl!(Dropout);
empty_npz_impl!(AvgPoolGlobal);
empty_npz_impl!(MaxPoolGlobal);
empty_npz_impl!(MinPoolGlobal);
empty_npz_impl!(Flatten2D);

impl<const N: usize> SaveToNpz for DropoutOneIn<N> {}
impl<const N: usize> LoadFromNpz for DropoutOneIn<N> {}

impl<const K: usize, const S: usize, const P: usize> SaveToNpz for AvgPool2D<K, S, P> {}
impl<const K: usize, const S: usize, const P: usize> LoadFromNpz for AvgPool2D<K, S, P> {}
impl<const K: usize, const S: usize, const P: usize> SaveToNpz for MaxPool2D<K, S, P> {}
impl<const K: usize, const S: usize, const P: usize> LoadFromNpz for MaxPool2D<K, S, P> {}
impl<const K: usize, const S: usize, const P: usize> SaveToNpz for MinPool2D<K, S, P> {}
impl<const K: usize, const S: usize, const P: usize> LoadFromNpz for MinPool2D<K, S, P> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::HasArrayType;
    use rand::thread_rng;
    use rand_distr::Standard;
    use tempfile::NamedTempFile;

    fn test_save_load<
        I: Tensor<Dtype = f32> + TensorCreator + Clone,
        M: Default + ResetParams + Module<I> + SaveToNpz + LoadFromNpz,
    >()
    where
        M::Output: HasArrayData,
        <M::Output as HasArrayType>::Array: std::fmt::Debug + PartialEq,
    {
        let mut rng = thread_rng();
        let x: I = TensorCreator::randn(&mut rng);
        let file = NamedTempFile::new().expect("failed to create tempfile");

        let mut saved: M = Default::default();
        let mut loaded: M = Default::default();

        saved.reset_params(&mut rng);
        let y = saved.forward(x.clone());

        assert_ne!(loaded.forward(x.clone()).data(), y.data());

        saved.save(file.path()).expect("");
        loaded.load(file.path()).expect("");

        assert_eq!(loaded.forward(x).data(), y.data());
    }

    #[test]
    fn test_batchnorm2d_save_load() {
        let mut rng = thread_rng();
        let x: Tensor3D<3, 4, 5> = TensorCreator::randn(&mut rng);
        let file = NamedTempFile::new().expect("failed to create tempfile");

        let mut saved: BatchNorm2D<3> = Default::default();
        let mut loaded: BatchNorm2D<3> = Default::default();

        saved.running_mean.randomize(&mut rng, &Standard);
        saved.running_var.randomize(&mut rng, &Standard);
        saved.scale.randomize(&mut rng, &Standard);
        saved.bias.randomize(&mut rng, &Standard);
        let y = saved.forward(x.clone());

        assert_ne!(loaded.forward(x.clone()).data(), y.data());

        saved.save(file.path()).expect("");
        loaded.load(file.path()).expect("");

        assert_eq!(loaded.forward(x).data(), y.data());
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_save_load_conv() {
        type T = Conv2D<2, 4, 3>;
        test_save_load::<Tensor3D<2, 8, 8>, T>();
    }

    #[test]
    fn test_save_load_generalized_residual() {
        type T = GeneralizedResidual<Linear<5, 5>, Linear<5, 5>>;
        test_save_load::<Tensor1D<5>, T>();
        test_save_load::<Tensor1D<5>, (T, T)>();
    }

    #[test]
    fn test_save_load_linear() {
        type T = Linear<5, 5>;
        test_save_load::<Tensor1D<5>, T>();
        test_save_load::<Tensor1D<5>, (T, T)>();
    }

    #[test]
    fn test_save_load_tuple() {
        type Model = (
            Linear<1, 2>,
            ReLU,
            Linear<2, 3>,
            (Dropout, Linear<3, 3>, Linear<3, 4>),
        );
        test_save_load::<Tensor1D<1>, Model>();
    }

    #[test]
    fn test_save_load_layer_norm() {
        type M = LayerNorm1D<3>;

        let mut rng = thread_rng();
        let x: Tensor1D<3> = TensorCreator::randn(&mut rng);
        let file = NamedTempFile::new().expect("failed to create tempfile");

        let mut saved: M = Default::default();
        let mut loaded: M = Default::default();

        saved.gamma.randomize(&mut rng, &Standard);
        saved.beta.randomize(&mut rng, &Standard);
        let y = saved.forward(x.clone());

        assert_ne!(loaded.forward(x.clone()).data(), y.data());

        saved.save(file.path()).expect("");
        loaded.load(file.path()).expect("");

        assert_eq!(loaded.forward(x).data(), y.data());
    }

    #[test]
    fn test_save_load_repeated() {
        type T = Repeated<Linear<3, 3>, 4>;
        test_save_load::<Tensor1D<3>, T>();
        test_save_load::<Tensor1D<3>, (T, T)>();
    }

    #[test]
    fn test_save_load_residual() {
        type T = Residual<Linear<5, 5>>;
        test_save_load::<Tensor1D<5>, T>();
        test_save_load::<Tensor1D<5>, (T, T)>();
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_save_load_mha() {
        let mut rng = thread_rng();

        let mut saved: MultiHeadAttention<12, 4> = Default::default();
        saved.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        saved.save(file.path()).expect("");

        let mut loaded: MultiHeadAttention<12, 4> = Default::default();
        loaded.load(file.path()).expect("");

        let q: Tensor3D<2, 3, 12> = TensorCreator::randn(&mut rng);
        let k: Tensor3D<2, 4, 12> = TensorCreator::randn(&mut rng);
        let v: Tensor3D<2, 4, 12> = TensorCreator::randn(&mut rng);
        let y1: Tensor3D<2, 3, 12, _> = saved.forward((q.clone(), k.clone(), v.clone()));
        let y2: Tensor3D<2, 3, 12, _> = loaded.forward((q.clone(), k.clone(), v.clone()));

        assert_eq!(y1.data(), y2.data());
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_save_load_transformer() {
        let mut rng = thread_rng();

        let mut saved: Transformer<16, 4, 3, 4, 8> = Default::default();
        saved.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        saved.save(file.path()).expect("");

        let mut loaded: Transformer<16, 4, 3, 4, 8> = Default::default();
        loaded.load(file.path()).expect("");

        let src: Tensor3D<4, 12, 16> = TensorCreator::randn(&mut rng);
        let tgt: Tensor3D<4, 6, 16> = TensorCreator::randn(&mut rng);

        let y1 = saved.forward_mut((src.clone(), tgt.clone()));
        let y2 = loaded.forward_mut((src.clone(), tgt.clone()));

        assert_eq!(y1.data(), y2.data());
    }
}
