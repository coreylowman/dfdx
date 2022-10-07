use crate::gradients::{CanUpdateWithGradients, GradientProvider, Tape, UnusedTensors};
use crate::prelude::*;
use rand::Rng;
use rand_distr::Uniform;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive, ZipWriter};

/// **Requires Nightly** Performs 2d convolutions on 3d and 4d images.
///
/// **Pytorch Equivalent**: `torch.nn.Conv2d`
///
/// Generics:
/// - `IN_CHAN`: The number of input channels in an image.
/// - `OUT_CHAN`: The number of channels in the output of the layer.
/// - `KERNEL_SIZE`: The size of the kernel applied to both width and height of the images.
/// - `STRIDE`: How far to move the kernel each step. Defaults to `1`
/// - `PADDING`: How much zero padding to add around the images. Defaults to `0`.
///
/// Examples:
/// ```rust
/// #![feature(generic_const_exprs)]
/// # use dfdx::prelude::*;
/// let m: Conv2D<16, 33, 3> = Default::default();
/// let _: Tensor3D<33, 30, 62> = m.forward(Tensor3D::<16, 32, 64>::zeros());
/// let _: Tensor4D<2, 33, 13, 12> = m.forward(Tensor4D::<2, 16, 15, 14>::zeros());
/// ```
#[derive(Default, Debug, Clone)]
pub struct Conv2D<
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize = 1,
    const PADDING: usize = 0,
> {
    pub weight: Tensor4D<OUT_CHAN, IN_CHAN, KERNEL_SIZE, KERNEL_SIZE>,
    pub bias: Tensor1D<OUT_CHAN>,
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize>
    CanUpdateWithGradients for Conv2D<I, O, K, S, P>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.weight.update(grads, unused);
        self.bias.update(grads, unused);
    }
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize> ResetParams
    for Conv2D<I, O, K, S, P>
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        let k = (I * K * K) as f32;
        let bound = 1.0 / k.sqrt();
        let dist = Uniform::new(-bound, bound);
        self.weight.randomize(rng, &dist);
        self.bias.randomize(rng, &dist);
    }
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize> SaveToNpz
    for Conv2D<I, O, K, S, P>
{
    /// Saves [Self::weight] to `{pre}weight.npy` and [Self::bias] to `{pre}bias.npy`
    /// using [npz_fwrite()].
    fn write<W: Write + Seek>(&self, pre: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        npz_fwrite(w, format!("{pre}weight.npy"), self.weight.data())?;
        npz_fwrite(w, format!("{pre}bias.npy"), self.bias.data())?;
        Ok(())
    }
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize> LoadFromNpz
    for Conv2D<I, O, K, S, P>
{
    /// Reads [Self::weight] from `{pre}weight.npy` and [Self::bias] from `{pre}bias.npy`
    /// using [npz_fread()].
    fn read<R: Read + Seek>(&mut self, pre: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        npz_fread(r, format!("{pre}weight.npy"), self.weight.mut_data())?;
        npz_fread(r, format!("{pre}bias.npy"), self.bias.mut_data())?;
        Ok(())
    }
}

impl<
        T: Tape,
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const H: usize,
        const W: usize,
    > Module<Tensor3D<I, H, W, T>> for Conv2D<I, O, K, S, P>
where
    [(); (W + 2 * P - K) / S + 1]:,
    [(); (H + 2 * P - K) / S + 1]:,
{
    type Output = Tensor3D<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>;

    fn forward(&self, x: Tensor3D<I, H, W, T>) -> Self::Output {
        x.conv2d::<O, K, S, P>(&self.weight, &self.bias)
    }
}

impl<
        T: Tape,
        const B: usize,
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const H: usize,
        const W: usize,
    > Module<Tensor4D<B, I, H, W, T>> for Conv2D<I, O, K, S, P>
where
    [(); (W + 2 * P - K) / S + 1]:,
    [(); (H + 2 * P - K) / S + 1]:,
{
    type Output = Tensor4D<B, O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>;

    fn forward(&self, x: Tensor4D<B, I, H, W, T>) -> Self::Output {
        x.conv2d::<O, K, S, P>(&self.weight, &self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    use std::fs::File;
    use tempfile::NamedTempFile;

    #[test]
    fn test_forward_3d_sizes() {
        type Img = Tensor3D<3, 10, 10>;
        let _: Tensor3D<2, 8, 8> = Conv2D::<3, 2, 3>::default().forward(Img::zeros());
        let _: Tensor3D<4, 8, 8> = Conv2D::<3, 4, 3>::default().forward(Img::zeros());
        let _: Tensor3D<4, 9, 9> = Conv2D::<3, 4, 2>::default().forward(Img::zeros());
        let _: Tensor3D<4, 7, 7> = Conv2D::<3, 4, 4>::default().forward(Img::zeros());
        let _: Tensor3D<2, 4, 4> = Conv2D::<3, 2, 3, 2>::default().forward(Img::zeros());
        let _: Tensor3D<2, 3, 3> = Conv2D::<3, 2, 3, 3>::default().forward(Img::zeros());
        let _: Tensor3D<2, 10, 10> = Conv2D::<3, 2, 3, 1, 1>::default().forward(Img::zeros());
        let _: Tensor3D<2, 12, 12> = Conv2D::<3, 2, 3, 1, 2>::default().forward(Img::zeros());
        let _: Tensor3D<2, 6, 6> = Conv2D::<3, 2, 3, 2, 2>::default().forward(Img::zeros());
    }

    #[test]
    fn test_forward_4d_sizes() {
        type Img = Tensor4D<5, 3, 10, 10>;
        let _: Tensor4D<5, 2, 8, 8> = Conv2D::<3, 2, 3>::default().forward(Img::zeros());
        let _: Tensor4D<5, 4, 8, 8> = Conv2D::<3, 4, 3>::default().forward(Img::zeros());
        let _: Tensor4D<5, 4, 9, 9> = Conv2D::<3, 4, 2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 4, 7, 7> = Conv2D::<3, 4, 4>::default().forward(Img::zeros());
        let _: Tensor4D<5, 2, 4, 4> = Conv2D::<3, 2, 3, 2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 2, 3, 3> = Conv2D::<3, 2, 3, 3>::default().forward(Img::zeros());
        let _: Tensor4D<5, 2, 10, 10> = Conv2D::<3, 2, 3, 1, 1>::default().forward(Img::zeros());
        let _: Tensor4D<5, 2, 12, 12> = Conv2D::<3, 2, 3, 1, 2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 2, 6, 6> = Conv2D::<3, 2, 3, 2, 2>::default().forward(Img::zeros());
    }

    #[test]
    fn test_2_conv_sizes() {
        type A = Conv2D<1, 2, 3>;
        type B = Conv2D<2, 4, 3>;
        let _: Tensor3D<4, 6, 6> = <(A, B)>::default().forward(Tensor3D::<1, 10, 10>::zeros());
    }

    #[test]
    fn test_3_conv_sizes() {
        type A = Conv2D<1, 2, 3>;
        type B = Conv2D<2, 4, 3>;
        type C = Conv2D<4, 1, 1, 1, 1>;

        type Img = Tensor3D<1, 10, 10>;
        let _: Tensor3D<1, 8, 8> = <(A, B, C)>::default().forward(Img::zeros());
    }

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
    fn test_conv_with_optimizer() {
        let mut rng = thread_rng();

        let mut m: Conv2D<2, 4, 3> = Default::default();
        m.reset_params(&mut rng);

        let weight_init = m.weight.clone();
        let bias_init = m.bias.clone();

        let mut opt: Sgd<_> = Default::default();
        let out = m.forward(Tensor4D::<8, 2, 28, 28>::randn(&mut rng).trace());
        let gradients = backward(out.square().mean());

        assert_ne!(gradients.ref_gradient(&m.weight), &[[[[0.0; 3]; 3]; 2]; 4]);
        assert_ne!(gradients.ref_gradient(&m.bias), &[0.0; 4]);

        opt.update(&mut m, gradients).expect("unused params");

        assert_ne!(weight_init.data(), m.weight.data());
        assert_ne!(bias_init.data(), m.bias.data());
    }
}
