use super::{npz_fread, npz_fwrite, LoadFromNpz, NpzError, SaveToNpz};
use super::{Module, ModuleMut, ResetParams};
use crate::arrays::{HasArrayData, HasAxes};
use crate::devices::{Cpu, FillElements};
use crate::gradients::*;
use crate::tensor::*;
use crate::tensor_ops::*;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive};

#[derive(Clone, Debug)]
pub struct BatchNorm2D<const C: usize> {
    pub gamma: Tensor1D<C>,
    pub beta: Tensor1D<C>,
    pub running_mean: Tensor1D<C>,
    pub running_var: Tensor1D<C>,
    pub epsilon: f32,
    pub momentum: f32,
}

impl<const C: usize> BatchNorm2D<C> {
    /// generic forward for inference
    fn infer_fwd<T: Tensor<Dtype = f32, NoTape = T>, Axes>(&self, x: T) -> T
    where
        Tensor1D<C>: BroadcastTo<T, Axes>,
    {
        // statistics for normalizing
        let std = (self.running_var.duplicate() + self.epsilon).sqrt();
        let mean = self.running_mean.duplicate();

        // normalize & affine
        let x = sub(x, &mean.broadcast());
        let x = div(x, &std.broadcast());
        let x = mul(x, &self.gamma.duplicate().broadcast());
        add(x, &self.beta.duplicate().broadcast())
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

        // record broadcast of gamma & beta - on tape
        let gamma: T = self.gamma.duplicate().put_tape(tape).broadcast();
        let (gamma, tape) = gamma.split_tape();
        let beta: T = self.beta.duplicate().put_tape(tape).broadcast();
        let (beta, tape) = beta.split_tape();

        // normalize & affine - on tape
        let x = sub(x.put_tape(tape), &mean);
        let x = div(x, &std);
        let x = mul(x, &gamma);
        add(x, &beta)
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
            gamma: TensorCreator::ones(),
            beta: TensorCreator::zeros(),
            running_mean: TensorCreator::zeros(),
            running_var: TensorCreator::ones(),
            epsilon: 1e-5,
            momentum: 0.1,
        }
    }
}

impl<const C: usize> ResetParams for BatchNorm2D<C> {
    fn reset_params<R: rand::Rng>(&mut self, _: &mut R) {
        Cpu::fill(self.gamma.mut_data(), &mut |v| *v = 1.0);
        Cpu::fill(self.beta.mut_data(), &mut |v| *v = 0.0);
        Cpu::fill(self.running_mean.mut_data(), &mut |v| *v = 0.0);
        Cpu::fill(self.running_var.mut_data(), &mut |v| *v = 1.0);
    }
}

impl<const C: usize> CanUpdateWithGradients for BatchNorm2D<C> {
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.gamma.update(grads, unused);
        self.beta.update(grads, unused);
    }
}

impl<const C: usize> SaveToNpz for BatchNorm2D<C> {
    fn write<W: Write + Seek>(&self, p: &str, w: &mut zip::ZipWriter<W>) -> ZipResult<()> {
        npz_fwrite(w, format!("{p}gamma.npy"), self.gamma.data())?;
        npz_fwrite(w, format!("{p}beta.npy"), self.beta.data())?;
        npz_fwrite(w, format!("{p}running_mean.npy"), self.running_mean.data())?;
        npz_fwrite(w, format!("{p}running_var.npy"), self.running_var.data())?;
        Ok(())
    }
}

impl<const C: usize> LoadFromNpz for BatchNorm2D<C> {
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        npz_fread(r, format!("{p}gamma.npy"), self.gamma.mut_data())?;
        npz_fread(r, format!("{p}beta.npy"), self.beta.mut_data())?;
        let mean = self.running_mean.mut_data();
        npz_fread(r, format!("{p}running_mean.npy"), mean)?;
        let var = self.running_mean.mut_data();
        npz_fread(r, format!("{p}running_var.npy"), var)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::{rngs::StdRng, SeedableRng};

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
        assert_close(g.ref_gradient(&bn.gamma), &[0.2506705, 0.4257624, 0.257648]);
        assert_close(g.ref_gradient(&bn.beta), &[0.4663894, 0.5239304, 0.4687197]);
        assert_eq!(
            g.ref_gradient(&x1),
            &[
                [[0.0030178577, 0.011973545], [0.0038383976, -0.018829815]],
                [[-0.0016367957, 0.024275035], [0.0092941, -0.03193234]],
                [[-0.015617318, 0.009291172], [0.0026013851, 0.0037247613]]
            ]
        );
    }

    #[test]
    fn test_batchnorm2d_4d_forward_mut() {
        todo!();
    }

    #[test]
    fn test_batchform2d_3d_repeated_forward_mut() {
        let mut rng = StdRng::seed_from_u64(12);

        let x1: Tensor3D<3, 2, 2> = TensorCreator::randn(&mut rng);
        let mut bn: BatchNorm2D<3> = Default::default();

        let _ = bn.forward_mut(x1.trace());
        todo!();
        let _ = bn.forward_mut(x1.trace());
        todo!();
        let _ = bn.forward_mut(x1.trace());
        todo!();
        let _ = bn.forward_mut(x1.trace());
        todo!();

        let m = bn.running_mean.clone();
        let v = bn.running_var.clone();

        let x2: Tensor3D<3, 2, 2> = TensorCreator::randn(&mut rng);
        let _ = bn.forward(x2);
        todo!();
        assert_eq!(bn.running_mean.data(), m.data());
        assert_eq!(bn.running_var.data(), v.data());
    }

    #[test]
    fn test_batchform2d_4d_repeated_forward_mut() {
        let mut rng = StdRng::seed_from_u64(13);

        let x1: Tensor4D<2, 3, 2, 2> = TensorCreator::randn(&mut rng);
        let mut bn: BatchNorm2D<3> = Default::default();

        let _ = bn.forward_mut(x1.trace());
        todo!();
        let _ = bn.forward_mut(x1.trace());
        todo!();
        let _ = bn.forward_mut(x1.trace());
        todo!();
        let _ = bn.forward_mut(x1.trace());
        todo!();

        let m = bn.running_mean.clone();
        let v = bn.running_var.clone();

        let x2: Tensor4D<4, 3, 2, 2> = TensorCreator::randn(&mut rng);
        let _ = bn.forward(x2);
        todo!();
        assert_eq!(bn.running_mean.data(), m.data());
        assert_eq!(bn.running_var.data(), v.data());
    }

    #[test]
    fn test_batchnorm2d_save_load() {
        todo!();
    }
}
