use super::{Module, ModuleMut, ResetParams};
use crate::arrays::{HasArrayData, HasAxes};
use crate::devices::{Cpu, FillElements};
use crate::{gradients::*, tensor::*, tensor_ops::*};

#[cfg(feature = "numpy")]
use super::{npz_fread, npz_fwrite, LoadFromNpz, NpzError, SaveToNpz};
#[cfg(feature = "numpy")]
use std::io::{Read, Seek, Write};
#[cfg(feature = "numpy")]
use zip::{result::ZipResult, ZipArchive};

/// Batch normalization for images as described in
/// [Batch Normalization: Accelerating Deep Network Training
/// by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
///
/// Generics:
///
/// - `C` the size of the spatial dimension to reduce. For 3d tensors this is the 0th
///   dimension. For 4d tensors, this is the 1st dimension.
///
/// # Training vs Inference
///
/// BatchNorm2D supports the following cases (see sections below for more details):
/// 1. **Training**: [ModuleMut] and [OwnedTape] on the input tensor
/// 2. **Inference**: [Module] and [NoneTape] on the input tensor.
///
/// *NOTE: ModuleMut/NoneTape, and Module/OwnedTape will fail to compile.*
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let bn: BatchNorm2D<3> = Default::default();
/// let _ = bn.forward(Tensor3D::<3, 2, 2>::zeros());
/// let _ = bn.forward(Tensor4D::<4, 3, 2, 2>::zeros());
/// ```
///
/// ### Training
/// - Running statistics: updated with momentum
/// - Normalization: calculated using batch stats
///
/// ### Inference
/// - Running statistics: **not** updated
/// - Normalization: calculated using running stats
#[derive(Clone, Debug)]
pub struct BatchNorm2D<const C: usize> {
    /// Scale for affine transform. Defaults to 1.0
    pub scale: Tensor1D<C>,
    /// Bias for affine transform. Defaults to 0.0
    pub bias: Tensor1D<C>,
    /// Spatial mean that is updated during training. Defaults to 0.0
    pub running_mean: Tensor1D<C>,
    /// Spatial variance that is updated during training. Defaults to 1.0
    pub running_var: Tensor1D<C>,
    /// Added to variance before taking sqrt for numerical stability. Defaults to 1e-5
    pub epsilon: f32,
    /// Controls exponential moving average of running stats.Defaults to 0.1
    ///
    /// `running_stat * (1.0 - momentum) + stat * momentum`.
    pub momentum: f32,
}

impl<const C: usize> BatchNorm2D<C> {
    /// generic forward for inference
    fn infer_fwd<T, Axes>(&self, x: T) -> T
    where
        T: Tensor<Dtype = f32, NoTape = T>,
        Tensor1D<C>: BroadcastTo<T, Axes>,
    {
        // statistics for normalizing
        let std = (self.running_var.duplicate() + self.epsilon).sqrt();
        let mean = self.running_mean.duplicate();

        // normalize & affine
        let x = sub(x, &mean.broadcast());
        let x = div(x, &std.broadcast());
        let x = mul(x, &self.scale.duplicate().broadcast());
        add(x, &self.bias.duplicate().broadcast())
    }

    fn train_fwd<T, Axes>(&mut self, x: T) -> T
    where
        T: Tensor<Dtype = f32, Tape = OwnedTape> + ReduceTo<Tensor1D<C, OwnedTape>, Axes>,
        T::Array: HasAxes<Axes>,
        Tensor1D<C, OwnedTape>: BroadcastTo<T, Axes>,
    {
        let (x, tape) = x.split_tape();

        // compute statistics for updating running stats later - on tape
        let (mean_t, tape): (Tensor1D<C>, _) = mean(x.duplicate().put_tape(tape)).split_tape();
        let (var_t, tape): (Tensor1D<C>, _) = var(x.duplicate().put_tape(tape)).split_tape();

        // update statistics since we are training - off tape
        self.running_mean = add(
            self.running_mean.duplicate() * (1.0 - self.momentum),
            &(mean_t.duplicate() * self.momentum),
        );
        let n = <T::Array as HasAxes<Axes>>::SIZE as f32;
        self.running_var = add(
            self.running_var.duplicate() * (1.0 - self.momentum),
            // NOTE: uses unbiased variance in running estimate
            &(var_t.duplicate() * (self.momentum * n / (n - 1.0))),
        );

        // statistics for normalizing - on tape
        let std: T = (var_t.put_tape(tape) + self.epsilon).sqrt().broadcast();
        let (std, tape) = std.split_tape();
        let mean: T = mean_t.put_tape(tape).broadcast();
        let (mean, tape) = mean.split_tape();

        // record broadcast of scale & bias - on tape
        let scale: T = self.scale.duplicate().put_tape(tape).broadcast();
        let (scale, tape) = scale.split_tape();
        let bias: T = self.bias.duplicate().put_tape(tape).broadcast();
        let (bias, tape) = bias.split_tape();

        // normalize & affine - on tape
        let x = sub(x.put_tape(tape), &mean);
        let x = div(x, &std);
        let x = mul(x, &scale);
        add(x, &bias)
    }
}

impl<const C: usize, const H: usize, const W: usize> Module<Tensor3D<C, H, W, NoneTape>>
    for BatchNorm2D<C>
{
    type Output = Tensor3D<C, H, W, NoneTape>;

    /// Inference 3d forward - does **not** update [Self::running_mean] and [Self::running_var]
    fn forward(&self, x: Tensor3D<C, H, W, NoneTape>) -> Self::Output {
        self.infer_fwd(x)
    }
}

impl<const B: usize, const C: usize, const H: usize, const W: usize>
    Module<Tensor4D<B, C, H, W, NoneTape>> for BatchNorm2D<C>
{
    type Output = Tensor4D<B, C, H, W, NoneTape>;

    /// Inference 4d forward - does **not** update [Self::running_mean] and [Self::running_var]
    fn forward(&self, x: Tensor4D<B, C, H, W, NoneTape>) -> Self::Output {
        self.infer_fwd(x)
    }
}

impl<const C: usize, const H: usize, const W: usize> ModuleMut<Tensor3D<C, H, W, OwnedTape>>
    for BatchNorm2D<C>
{
    type Output = Tensor3D<C, H, W, OwnedTape>;

    /// Training 3d forward - updates [Self::running_mean] and [Self::running_var]
    fn forward_mut(&mut self, x: Tensor3D<C, H, W, OwnedTape>) -> Self::Output {
        self.train_fwd(x)
    }
}

impl<const B: usize, const C: usize, const H: usize, const W: usize>
    ModuleMut<Tensor4D<B, C, H, W, OwnedTape>> for BatchNorm2D<C>
{
    type Output = Tensor4D<B, C, H, W, OwnedTape>;

    /// Training 4d forward - updates [Self::running_mean] and [Self::running_var]
    fn forward_mut(&mut self, x: Tensor4D<B, C, H, W, OwnedTape>) -> Self::Output {
        self.train_fwd(x)
    }
}

impl<const C: usize> Default for BatchNorm2D<C> {
    fn default() -> Self {
        Self {
            scale: TensorCreator::ones(),
            bias: TensorCreator::zeros(),
            running_mean: TensorCreator::zeros(),
            running_var: TensorCreator::ones(),
            epsilon: 1e-5,
            momentum: 0.1,
        }
    }
}

impl<const C: usize> ResetParams for BatchNorm2D<C> {
    fn reset_params<R: rand::Rng>(&mut self, _: &mut R) {
        Cpu::fill(self.scale.mut_data(), &mut |v| *v = 1.0);
        Cpu::fill(self.bias.mut_data(), &mut |v| *v = 0.0);
        Cpu::fill(self.running_mean.mut_data(), &mut |v| *v = 0.0);
        Cpu::fill(self.running_var.mut_data(), &mut |v| *v = 1.0);
    }
}

impl<const C: usize> CanUpdateWithGradients for BatchNorm2D<C> {
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.scale.update(grads, unused);
        self.bias.update(grads, unused);
    }
}

#[cfg(feature = "numpy")]
impl<const C: usize> SaveToNpz for BatchNorm2D<C> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut zip::ZipWriter<W>) -> ZipResult<()> {
        npz_fwrite(w, format!("{p}scale.npy"), self.scale.data())?;
        npz_fwrite(w, format!("{p}bias.npy"), self.bias.data())?;
        npz_fwrite(w, format!("{p}running_mean.npy"), self.running_mean.data())?;
        npz_fwrite(w, format!("{p}running_var.npy"), self.running_var.data())?;
        Ok(())
    }
}

#[cfg(feature = "numpy")]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nn::tests::SimpleGradients, tests::assert_close};
    use rand::{rngs::StdRng, SeedableRng};
    use tempfile::NamedTempFile;

    #[test]
    fn test_batchnorm2d_3d_forward_mut() {
        let mut rng = StdRng::seed_from_u64(0);
        let x1: Tensor3D<3, 2, 2> = TensorCreator::randn(&mut rng);
        let mut bn: BatchNorm2D<3> = Default::default();

        let y1 = bn.forward_mut(x1.trace());
        assert_close(
            y1.data(),
            &[
                [[0.66747534, 0.77682495], [-1.698878, 0.25457793]],
                [[-0.89111614, 1.2611268], [-1.0644908, 0.69448]],
                [[0.19064833, 0.80228466], [0.6924452, -1.6853783]],
            ],
        );

        let g = backward(y1.exp().mean());
        assert_close(bn.running_mean.data(), &[-0.0175438, -0.0214163, 0.0268384]);
        assert_close(bn.running_var.data(), &[1.1361228, 1.0889612, 1.3478994]);
        assert_close(g.ref_gradient(&bn.scale), &[0.2506705, 0.4257624, 0.257648]);
        assert_close(g.ref_gradient(&bn.bias), &[0.4663894, 0.5239304, 0.4687197]);
        assert_close(
            g.ref_gradient(&x1),
            &[
                [[0.0030178577, 0.011973545], [0.0038383976, -0.018829815]],
                [[-0.0016367957, 0.024275035], [0.0092941, -0.03193234]],
                [[-0.015617318, 0.009291172], [0.0026013851, 0.0037247613]],
            ],
        );
    }

    #[test]
    fn test_batchnorm2d_4d_forward_mut() {
        let mut rng = StdRng::seed_from_u64(2);
        let x1: Tensor4D<2, 2, 2, 3> = TensorCreator::randn(&mut rng);
        let mut bn: BatchNorm2D<2> = Default::default();

        let y1 = bn.forward_mut(x1.trace());
        #[rustfmt::skip]
        assert_close(
            y1.data(),
            &[
                [[[-0.93348885, -2.1979978, 0.19754872],[0.29159376, -0.6282544, -1.0415624]], [[1.1156346, 0.89029306, -1.1608727],[-0.73874927, 0.13254784, -0.77676374]]],
                [[[0.60655713, 0.62703574, 0.12648833],[1.5577206, 0.18830705, 1.2060523]],[[0.37415895, -0.9069047, -0.9519587],[-0.02608296, 2.3435123, -0.2948149]]],
            ],
        );

        let g = backward(y1.exp().mean());
        assert_close(bn.running_mean.data(), &[-0.02424082, 0.00407672]);
        assert_close(bn.running_var.data(), &[0.9676103, 1.0458221]);
        assert_close(g.ref_gradient(&bn.scale), &[0.5582906, 1.1929206]);
        assert_close(g.ref_gradient(&bn.bias), &[0.7535024, 0.92750454]);
        #[rustfmt::skip]
        assert_close(
            g.ref_gradient(&x1),
            &[
                [[[-0.00378475, 0.05601016, -0.02694868],[-0.02614748, -0.01439525, 0.00047035]],[[-0.05280511, -0.05561727, 0.04425058],[0.01388359, -0.03710236, 0.01651]]],
                [[[-0.01853323, -0.01773504, -0.02717264],[0.0794776, -0.02699574, 0.02575465]],[[-0.04663141, 0.02567738, 0.0289102],[-0.0294986, 0.10708933, -0.01466625]]],
            ],
        );
    }

    #[test]
    fn test_batchform2d_3d_repeated_forward_mut() {
        let mut rng = StdRng::seed_from_u64(12);

        let x1: Tensor3D<3, 4, 5> = TensorCreator::randn(&mut rng);
        let mut bn: BatchNorm2D<3> = Default::default();

        let _ = bn.forward_mut(x1.trace());
        assert_close(bn.running_mean.data(), &[0.0083191, -0.0370511, -0.0079481]);
        assert_close(bn.running_var.data(), &[1.0344709, 0.9340682, 1.0266376]);

        let _ = bn.forward_mut(x1.trace());
        assert_close(bn.running_mean.data(), &[0.0158063, -0.0703971, -0.0151013]);
        assert_close(bn.running_var.data(), &[1.0654946, 0.87472963, 1.0506116]);

        let _ = bn.forward_mut(x1.trace());
        assert_close(bn.running_mean.data(), &[0.0225448, -0.1004085, -0.0215393]);
        assert_close(bn.running_var.data(), &[1.093416, 0.8213248, 1.0721881]);

        let _ = bn.forward_mut(x1.trace());
        assert_close(bn.running_mean.data(), &[0.0286095, -0.1274188, -0.0273335]);
        assert_close(bn.running_var.data(), &[1.1185452, 0.7732605, 1.0916069]);

        let m = bn.running_mean.clone();
        let v = bn.running_var.clone();

        let x2: Tensor3D<3, 2, 2> = TensorCreator::randn(&mut rng);
        let y2 = bn.forward(x2);
        // running stats shouldn't have been updated
        assert_eq!(bn.running_mean.data(), m.data());
        assert_eq!(bn.running_var.data(), v.data());
        assert_close(
            y2.data(),
            &[
                [[0.0897828, -0.01880704], [-0.55082226, -0.50515544]],
                [[0.13778551, 0.25317147], [-1.2689502, 0.61595416]],
                [[0.73018146, 0.3243845], [-1.1041277, 0.38778353]],
            ],
        );
    }

    #[cfg(feature = "numpy")]
    #[test]
    fn test_batchnorm2d_save_load() {
        let mut rng = StdRng::seed_from_u64(13);
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
}
