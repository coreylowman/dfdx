use crate::prelude::*;

/// Batch normalization for sequences as described in
/// [Batch Normalization: Accelerating Deep Network Training
/// by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
///
/// Generics:
///
/// - `C` the size of the dimension to reduce. Both for 2d tensors (of the form <BATCH_SIZE, C>)
///   as well as 3d tensors (of the form <BATCH_SIZE, C, SEQUENCE_LENGTH>), this is the 1st dimension.
///
/// # Training vs Inference
///
/// BatchNorm1D supports the following cases (see sections below for more details):
/// 1. **Training**: [crate::nn::Module::forward_mut()] and [OwnedTape] on the input tensor
/// 2. **Inference**: [crate::nn::Module::forward()] and [NoneTape] on the input tensor.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx::*;
/// # let dev: Cpu = Default::default();
/// type Model = BatchNorm1DConstConfig<3>;
/// let bn = dev.build_module::<f32>(Model::default());
/// let _ = bn.forward(dev.zeros::<Rank2<4, 3>>());
/// let _ = bn.forward(dev.zeros::<Rank3<4, 3, 2>>());
/// ```
///
/// ### Training
/// - Running statistics: updated with momentum
/// - Normalization: calculated using batch stats
///
/// ### Inference
/// - Running statistics: **not** updated
/// - Normalization: calculated using running stats
#[derive(Default, Clone, Copy, Debug)]
#[repr(transparent)]
pub struct BatchNorm1DConfig<C: Dim>(pub C);

/// Compile time sugar alias around [BatchNorm1DConfig]
pub type BatchNorm1DConstConfig<const C: usize> = BatchNorm1DConfig<Const<C>>;

impl<C: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for BatchNorm1DConfig<C> {
    type Built = BatchNorm1D<C, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(BatchNorm1D {
            scale: device.try_ones_like(&(self.0,))?,
            bias: device.try_zeros_like(&(self.0,))?,
            running_mean: device.try_zeros_like(&(self.0,))?,
            running_var: device.try_ones_like(&(self.0,))?,
            epsilon: 1e-5,
            momentum: 0.1,
        })
    }
}

/// See [BatchNorm1DConfig].
#[derive(Clone, Debug, UpdateParams, ZeroGrads, WithGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct BatchNorm1D<C: Dim, Elem: Dtype, Dev: Device<Elem>> {
    /// Scale for affine transform. Defaults to 1.0
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub scale: Tensor<(C,), Elem, Dev>,
    /// Bias for affine transform. Defaults to 0.0
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub bias: Tensor<(C,), Elem, Dev>,
    /// Spatial mean that is updated during training. Defaults to 0.0
    #[cfg_attr(feature = "safetensors", serialize)]
    pub running_mean: Tensor<(C,), Elem, Dev>,
    /// Spatial variance that is updated during training. Defaults to 1.0
    #[cfg_attr(feature = "safetensors", serialize)]
    pub running_var: Tensor<(C,), Elem, Dev>,
    /// Added to variance before taking sqrt for numerical stability. Defaults to 1e-5
    #[cfg_attr(feature = "safetensors", serialize)]
    pub epsilon: f64,
    /// Controls exponential moving average of running stats. Defaults to 0.1
    ///
    /// `running_stat * (1.0 - momentum) + stat * momentum`.
    #[cfg_attr(feature = "safetensors", serialize)]
    pub momentum: f64,
}

impl<C: Dim, E: Dtype, D: Device<E>> crate::nn::ResetParams<E, D> for BatchNorm1D<C, E, D> {
    fn try_reset_params(&mut self) -> Result<(), crate::tensor::Error> {
        self.scale.try_fill_with_ones()?;
        self.bias.try_fill_with_zeros()?;
        self.running_mean.try_fill_with_zeros()?;
        self.running_var.try_fill_with_ones()
    }
}

impl<B: Dim, C: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::nn::Module<Tensor<(B, C), E, D, T>> for BatchNorm1D<C, E, D>
{
    type Output = Tensor<(B, C), E, D, T>;
    fn try_forward(&self, x: Tensor<(B, C), E, D, T>) -> Result<Self::Output, Error> {
        assert!(!T::OWNS_TAPE);
        self.infer_fwd(x)
    }
    fn try_forward_mut(&mut self, x: Tensor<(B, C), E, D, T>) -> Result<Self::Output, Error> {
        assert!(T::OWNS_TAPE);
        self.train_fwd(x)
    }
}

impl<B: Dim, C: Dim, L: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::nn::Module<Tensor<(B, C, L), E, D, T>> for BatchNorm1D<C, E, D>
{
    type Output = Tensor<(B, C, L), E, D, T>;
    fn try_forward(&self, x: Tensor<(B, C, L), E, D, T>) -> Result<Self::Output, Error> {
        assert!(!T::OWNS_TAPE);
        self.infer_fwd(x)
    }
    fn try_forward_mut(&mut self, x: Tensor<(B, C, L), E, D, T>) -> Result<Self::Output, Error> {
        assert!(T::OWNS_TAPE);
        self.train_fwd(x)
    }
}

impl<C: Dim, E: Dtype, D: Device<E>> BatchNorm1D<C, E, D> {
    /// generic batchnorm forward for training
    fn train_fwd<S: Shape, T: Tape<E, D>, Ax: Axes>(
        &mut self,
        x: Tensor<S, E, D, T>,
    ) -> Result<Tensor<S, E, D, T>, crate::tensor::Error>
    where
        S: HasAxes<Ax> + ReduceShapeTo<(C,), Ax>,
    {
        let n = <S as HasAxes<Ax>>::size(x.shape()) as f64;
        let shape = *x.shape();

        // compute statistics for updating running stats later - on tape
        let mean_chan = x.retaped::<T>().try_mean::<(C,), _>()?;

        // update statistics since we are training - off tape
        self.running_mean
            .try_axpy(1.0 - self.momentum, &mean_chan, self.momentum)?;

        let centered = x.try_sub(mean_chan.try_broadcast_like(&shape)?)?;

        let var_chan = centered
            .retaped::<T>()
            .try_square()?
            .try_mean::<(C,), _>()?;

        // NOTE: uses unbiased variance in running estimate
        self.running_var.try_axpy(
            1.0 - self.momentum,
            &var_chan,
            self.momentum * n / (n - 1.0),
        )?;

        // statistics for normalizing - on tape
        let std = var_chan.try_add(self.epsilon)?.try_sqrt()?;

        // record broadcast of scale & bias - on tape
        let scale = self
            .scale
            .retaped::<T>()
            .try_div(std)?
            .try_broadcast_like(&shape)?;
        let bias = self.bias.retaped::<T>().try_broadcast_like(&shape)?;

        // normalize & affine - on tape
        centered.try_mul(scale)?.try_add(bias)
    }

    /// generic batchnorm forward for inference
    pub fn infer_fwd<S: Shape, T: Tape<E, D>, Ax: Axes>(
        &self,
        x: Tensor<S, E, D, T>,
    ) -> Result<Tensor<S, E, D, T>, crate::tensor::Error>
    where
        (C,): BroadcastShapeTo<S, Ax>,
    {
        let shape = *x.shape();

        // statistics for normalizing
        let std = self.running_var.clone().try_add(self.epsilon)?.try_sqrt()?;

        let scale = self
            .scale
            .clone()
            .try_div(std)?
            .try_broadcast_like(&shape)?;

        // normalize & affine
        let x = x.try_sub(self.running_mean.clone().try_broadcast_like(&shape)?)?;
        let x = x.try_mul(scale)?;
        x.try_add(self.bias.clone().try_broadcast_like(&shape)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_batchnorm1d_2d_forward_mut() {
        let dev = TestDevice::seed_from_u64(0);

        let x1: Tensor<Rank2<3, 2>, TestDtype, _> = dev.sample(rand_distr::StandardNormal);
        let mut bn = dev.build_module::<TestDtype>(BatchNorm1DConstConfig::<2>::default());

        let y1 = bn.forward_mut(x1.leaky_trace());
        assert_close_to_literal!(
            y1,
            [
                [1.3168651, 0.19157785],
                [-1.1049646, -1.3092154],
                [-0.21190044, 1.1176374],
            ]
        );

        let g = y1.exp().mean().backward();
        assert_close_to_literal!(bn.running_mean, [-0.09994803, 0.07696156]);
        assert_close_to_literal!(bn.running_var, [1.1536077, 0.9321649]);
        assert_close_to_literal!(g.get(&bn.scale), [0.72945416, 0.5493023]);
        assert_close_to_literal!(g.get(&bn.bias), [0.8119954, 0.7564688]);
        assert_close_to_literal!(
            g.get(&x1),
            [
                [0.023908734, -0.18436226],
                [0.040923715, 0.0703277],
                [-0.06483248, 0.11403453],
            ]
        );
    }

    #[test]
    fn test_batchnorm1d_3d_forward_mut() {
        const BATCH_SIZE: usize = 3;
        const DIMENSION: usize = 2;
        const LENGTH: usize = 2;
        let dev = TestDevice::seed_from_u64(0);

        let x1: Tensor<Rank3<BATCH_SIZE, DIMENSION, LENGTH>, TestDtype, _> =
            dev.sample(rand_distr::StandardNormal);
        let mut bn = dev.build_module::<TestDtype>(BatchNorm1DConstConfig::<DIMENSION>::default());

        let y1 = bn.forward_mut(x1.leaky_trace());
        assert_close_to_literal!(
            y1,
            [
                [[0.059494145, 0.21366562], [-1.0539212, 0.5588659]],
                [[-2.0465322, 0.6680055], [-0.46153978, 0.8375814]],
                [[-0.041158404, 1.1465254], [1.411404, -1.2923905]],
            ]
        );

        let g = y1.exp().mean().backward();
        assert_close_to_literal!(bn.running_mean, [0.065665804, -0.07374697]);
        assert_close_to_literal!(bn.running_var, [1.0069065, 1.2117702]);
        assert_close_to_literal!(g.get(&bn.scale), [0.4112549, 0.6407272]);
        assert_close_to_literal!(g.get(&bn.bias), [0.7071625, 0.78455544]);
        assert_close_to_literal!(
            g.get(&x1),
            [
                [[-0.035488494, -0.031065114], [0.0067214966, -0.02774144]],
                [[0.035152107, -0.0011850521], [-0.017958358, -0.017146945]],
                [[-0.03715139, 0.0697379], [0.037428252, 0.018696927]],
            ]
        );
    }

    #[test]
    fn test_batchnorm1d_update() {
        const BATCH_SIZE: usize = 3;
        const DIMENSION: usize = 4;
        let dev: TestDevice = Default::default();

        let x1: Tensor<Rank2<BATCH_SIZE, DIMENSION>, TestDtype, _> = dev.sample_normal();
        let mut bn = dev.build_module::<TestDtype>(BatchNorm1DConstConfig::<DIMENSION>::default());
        let y = bn.forward_mut(x1.leaky_trace());
        let g = y.square().mean().backward();

        let mut opt = crate::nn::optim::Sgd::new(&bn, Default::default());
        opt.update(&mut bn, &g).expect("");
    }
}
