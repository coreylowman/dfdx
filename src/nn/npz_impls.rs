use super::npz::{npz_fread, npz_fwrite, LoadFromNpz, SaveToNpz};
use crate::prelude::*;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive, ZipWriter};

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

#[cfg(feature = "nightly")]
impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize> SaveToNpz
    for Conv2D<I, O, K, S, P>
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        npz_fwrite(w, format!("{p}weight.npy"), self.weight.data())?;
        npz_fwrite(w, format!("{p}bias.npy"), self.bias.data())?;
        Ok(())
    }
}

#[cfg(feature = "nightly")]
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
    fn write<W: Write + Seek>(&self, base: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        $(self.$idx.write(&format!("{}{}.", base, $idx), w)?;)+
        Ok(())
    }
}

impl<$($name: LoadFromNpz),+> LoadFromNpz for ($($name,)+) {
    fn read<R: Read + Seek>(&mut self, base: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        $(self.$idx.read(&format!("{}{}.", base, $idx), r)?;)+
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
        self.0.write(p, w)
    }
}

impl<F: LoadFromNpz> LoadFromNpz for Residual<F> {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.0.read(p, r)
    }
}

impl<T: SaveToNpz> SaveToNpz for SplitInto<T> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.0.write(p, w)
    }
}

impl<T: LoadFromNpz> LoadFromNpz for SplitInto<T> {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.0.read(p, r)
    }
}

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const F: usize, const L: usize> SaveToNpz
    for TransformerDecoder<M, H, F, L>
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.0.write(p, w)
    }
}

#[cfg(feature = "nightly")]
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

#[cfg(feature = "nightly")]
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

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const F: usize, const L: usize> LoadFromNpz
    for TransformerDecoder<M, H, F, L>
{
    fn read<R: Read + Seek>(&mut self, pre: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.0.read(pre, r)
    }
}

#[cfg(feature = "nightly")]
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

#[cfg(feature = "nightly")]
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

#[cfg(feature = "nightly")]
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

#[cfg(feature = "nightly")]
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

#[cfg(feature = "nightly")]
impl<const M: usize, const H: usize, const E: usize, const D: usize, const F: usize> SaveToNpz
    for Transformer<M, H, E, D, F>
{
    fn write<W: Write + Seek>(&self, p: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.encoder.write(&format!("{p}encoder."), w)?;
        self.decoder.write(&format!("{p}decoder."), w)?;
        Ok(())
    }
}

#[cfg(feature = "nightly")]
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
#[cfg(feature = "nightly")]
empty_npz_impl!(FlattenImage);

impl<const N: usize> SaveToNpz for DropoutOneIn<N> {}
impl<const N: usize> LoadFromNpz for DropoutOneIn<N> {}

#[cfg(feature = "nightly")]
impl<const K: usize, const S: usize, const P: usize> SaveToNpz for AvgPool2D<K, S, P> {}
#[cfg(feature = "nightly")]
impl<const K: usize, const S: usize, const P: usize> LoadFromNpz for AvgPool2D<K, S, P> {}

#[cfg(feature = "nightly")]
impl<const K: usize, const S: usize, const P: usize> SaveToNpz for MaxPool2D<K, S, P> {}
#[cfg(feature = "nightly")]
impl<const K: usize, const S: usize, const P: usize> LoadFromNpz for MaxPool2D<K, S, P> {}

#[cfg(feature = "nightly")]
impl<const K: usize, const S: usize, const P: usize> SaveToNpz for MinPool2D<K, S, P> {}
#[cfg(feature = "nightly")]
impl<const K: usize, const S: usize, const P: usize> LoadFromNpz for MinPool2D<K, S, P> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradients::*;
    use crate::nn::tests::SimpleGradients;
    use rand::thread_rng;
    use rand_distr::Standard;
    use std::fs::File;
    use tempfile::NamedTempFile;

    #[test]
    fn test_batchnorm2d_save_load() {
        let mut rng = thread_rng();
        let mut bn: BatchNorm2D<3> = Default::default();

        assert_eq!(bn.running_mean.data(), &[0.0; 3]);
        assert_eq!(bn.running_var.data(), &[1.0; 3]);
        assert_eq!(bn.scale.data(), &[1.0; 3]);
        assert_eq!(bn.bias.data(), &[0.0; 3]);

        let x1: Tensor3D<3, 4, 5> = TensorCreator::randn(&mut rng);
        let g = backward(bn.forward_mut(x1.trace()).exp().mean());
        bn.update(&mut SimpleGradients(g), &mut Default::default());

        assert_ne!(bn.running_mean.data(), &[0.0; 3]);
        assert_ne!(bn.running_var.data(), &[1.0; 3]);
        assert_ne!(bn.scale.data(), &[1.0; 3]);
        assert_ne!(bn.bias.data(), &[0.0; 3]);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(bn.save(file.path().to_str().unwrap()).is_ok());

        let mut loaded: BatchNorm2D<3> = Default::default();
        assert!(loaded.load(file.path().to_str().unwrap()).is_ok());
        assert_eq!(loaded.scale.data(), bn.scale.data());
        assert_eq!(loaded.bias.data(), bn.bias.data());
        assert_eq!(loaded.running_mean.data(), bn.running_mean.data());
        assert_eq!(loaded.running_var.data(), bn.running_var.data());
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_save_conv2d() {
        let model: Conv2D<2, 4, 3> = Default::default();
        let file = NamedTempFile::new().expect("failed to create tempfile");
        model
            .save(file.path().to_str().unwrap())
            .expect("failed to save model");
        let f = File::open(file.path()).expect("failed to open resulting file");
        let mut zip = ZipArchive::new(f).expect("failed to create zip archive from file");
        {
            let weight_file = zip
                .by_name("weight.npy")
                .expect("failed to find weight.npy file");
            assert!(weight_file.size() > 0);
        }
        {
            let bias_file = zip
                .by_name("bias.npy")
                .expect("failed to find bias.npy file");
            assert!(bias_file.size() > 0);
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_load_conv() {
        let mut rng = thread_rng();
        let mut saved_model: Conv2D<2, 4, 3> = Default::default();
        saved_model.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model.save(file.path().to_str().unwrap()).is_ok());

        let mut loaded_model: Conv2D<2, 4, 3> = Default::default();
        assert!(loaded_model.weight.data() != saved_model.weight.data());
        assert!(loaded_model.bias.data() != saved_model.bias.data());

        assert!(loaded_model.load(file.path().to_str().unwrap()).is_ok());
        assert_eq!(loaded_model.weight.data(), saved_model.weight.data());
        assert_eq!(loaded_model.bias.data(), saved_model.bias.data());
    }

    #[test]
    fn test_save_load_generalized_residual() {
        let mut rng = thread_rng();
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

    #[test]
    fn test_save_linear() {
        let model: Linear<5, 3> = Default::default();
        let file = NamedTempFile::new().expect("failed to create tempfile");
        model
            .save(file.path().to_str().unwrap())
            .expect("failed to save model");
        let f = File::open(file.path()).expect("failed to open resulting file");
        let mut zip = ZipArchive::new(f).expect("failed to create zip archive from file");
        {
            let weight_file = zip
                .by_name("weight.npy")
                .expect("failed to find weight.npy file");
            assert!(weight_file.size() > 0);
        }
        {
            let bias_file = zip
                .by_name("bias.npy")
                .expect("failed to find bias.npy file");
            assert!(bias_file.size() > 0);
        }
    }

    #[test]
    fn test_load_linear() {
        let mut rng = thread_rng();
        let mut saved_model: Linear<5, 3> = Default::default();
        saved_model.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model.save(file.path().to_str().unwrap()).is_ok());

        let mut loaded_model: Linear<5, 3> = Default::default();
        assert!(loaded_model.weight.data() != saved_model.weight.data());
        assert!(loaded_model.bias.data() != saved_model.bias.data());

        assert!(loaded_model.load(file.path().to_str().unwrap()).is_ok());
        assert_eq!(loaded_model.weight.data(), saved_model.weight.data());
        assert_eq!(loaded_model.bias.data(), saved_model.bias.data());
    }

    #[test]
    fn test_save_tuple() {
        let model: (
            Linear<1, 2>,
            ReLU,
            Linear<2, 3>,
            (Dropout, Linear<1, 2>, Linear<3, 4>),
        ) = Default::default();
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
                "2.bias.npy",
                "2.weight.npy",
                "3.1.bias.npy",
                "3.1.weight.npy",
                "3.2.bias.npy",
                "3.2.weight.npy",
            ]
        );
    }

    #[test]
    fn test_load_tuple() {
        type Model = (
            Linear<1, 2>,
            ReLU,
            Linear<2, 3>,
            (Dropout, Linear<1, 2>, Linear<3, 4>),
        );

        let mut rng = thread_rng();
        let mut saved_model: Model = Default::default();
        saved_model.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model.save(file.path().to_str().unwrap()).is_ok());

        let mut loaded_model: Model = Default::default();
        assert!(loaded_model.load(file.path().to_str().unwrap()).is_ok());
        assert_eq!(loaded_model.0.weight.data(), saved_model.0.weight.data());
        assert_eq!(loaded_model.0.bias.data(), saved_model.0.bias.data());
        assert_eq!(loaded_model.2.weight.data(), saved_model.2.weight.data());
        assert_eq!(loaded_model.2.bias.data(), saved_model.2.bias.data());
        assert_eq!(
            loaded_model.3 .1.weight.data(),
            saved_model.3 .1.weight.data()
        );
        assert_eq!(loaded_model.3 .1.bias.data(), saved_model.3 .1.bias.data());

        assert_eq!(
            loaded_model.3 .2.weight.data(),
            saved_model.3 .2.weight.data()
        );
        assert_eq!(loaded_model.3 .2.bias.data(), saved_model.3 .2.bias.data());
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
        let mut rng = thread_rng();
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

        let mut rng = thread_rng();
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
    fn test_save_load_residual() {
        let mut rng = thread_rng();
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

    #[cfg(feature = "nightly")]
    #[test]
    fn test_save_and_load() {
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
    fn test_save_load() {
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
