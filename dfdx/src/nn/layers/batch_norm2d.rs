use crate::prelude::*;

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
/// 1. **Training**: [crate::nn::Module::forward_mut()] and [OwnedTape] on the input tensor
/// 2. **Inference**: [crate::nn::Module::forward()] and [NoneTape] on the input tensor.
///
/// *NOTE: ModuleMut/NoneTape, and Module/OwnedTape will fail to compile.*
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx::*;
/// # let dev: Cpu = Default::default();
/// type Model = BatchNorm2DConstConfig<3>;
/// let bn = dev.build_module::<f32>(Model::default());
/// let _ = bn.forward(dev.zeros::<Rank3<3, 2, 2>>());
/// let _ = bn.forward(dev.zeros::<Rank4<4, 3, 2, 2>>());
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
pub struct BatchNorm2DConfig<C: Dim>(pub C);

/// Compile time sugar alias around [BatchNorm2DConfig]
pub type BatchNorm2DConstConfig<const C: usize> = BatchNorm2DConfig<Const<C>>;

impl<C: Dim, E: Dtype, D: Device<E>> crate::nn::BuildOnDevice<E, D> for BatchNorm2DConfig<C> {
    type Built = BatchNorm2D<C, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(BatchNorm2D {
            scale: device.try_ones_like(&(self.0,))?,
            bias: device.try_zeros_like(&(self.0,))?,
            running_mean: device.try_zeros_like(&(self.0,))?,
            running_var: device.try_ones_like(&(self.0,))?,
            epsilon: 1e-5,
            momentum: 0.1,
        })
    }
}

/// See [BatchNorm2DConfig]
#[derive(Clone, Debug, UpdateParams, ZeroGrads, WithGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct BatchNorm2D<C: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub scale: Tensor<(C,), Elem, Dev>,
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub bias: Tensor<(C,), Elem, Dev>,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub running_mean: Tensor<(C,), Elem, Dev>,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub running_var: Tensor<(C,), Elem, Dev>,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub epsilon: f64,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub momentum: f64,
}

impl<C: Dim, E: Dtype, D: Device<E>> crate::nn::ResetParams<E, D> for BatchNorm2D<C, E, D> {
    fn try_reset_params(&mut self) -> Result<(), crate::tensor::Error> {
        self.scale.try_fill_with_ones()?;
        self.bias.try_fill_with_zeros()?;
        self.running_mean.try_fill_with_zeros()?;
        self.running_var.try_fill_with_ones()
    }
}

impl<C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::nn::Module<Tensor<(C, H, W), E, D, T>> for BatchNorm2D<C, E, D>
{
    type Output = Tensor<(C, H, W), E, D, T>;
    fn try_forward(&self, x: Tensor<(C, H, W), E, D, T>) -> Result<Self::Output, Error> {
        assert!(!T::OWNS_TAPE);
        self.infer_fwd(x)
    }
    fn try_forward_mut(&mut self, x: Tensor<(C, H, W), E, D, T>) -> Result<Self::Output, Error> {
        assert!(T::OWNS_TAPE);
        self.train_fwd(x)
    }
}

impl<Batch: Dim, C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::nn::Module<Tensor<(Batch, C, H, W), E, D, T>> for BatchNorm2D<C, E, D>
{
    type Output = Tensor<(Batch, C, H, W), E, D, T>;
    fn try_forward(&self, x: Tensor<(Batch, C, H, W), E, D, T>) -> Result<Self::Output, Error> {
        assert!(!T::OWNS_TAPE);
        self.infer_fwd(x)
    }
    fn try_forward_mut(
        &mut self,
        x: Tensor<(Batch, C, H, W), E, D, T>,
    ) -> Result<Self::Output, Error> {
        assert!(T::OWNS_TAPE);
        self.train_fwd(x)
    }
}

impl<C: Dim, E: Dtype, D: Device<E>> BatchNorm2D<C, E, D> {
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
    fn test_batchnorm2d_3d_forward_mut() {
        let dev = TestDevice::seed_from_u64(0);

        let x1: Tensor<Rank3<3, 2, 2>, TestDtype, _> = dev.sample(rand_distr::StandardNormal);
        let mut bn = dev.build_module::<TestDtype>(BatchNorm2DConstConfig::<3>::default());

        let y1 = bn.forward_mut(x1.leaky_trace());
        assert_close_to_literal!(
            y1,
            [
                [[0.66747534, 0.77682495], [-1.698878, 0.25457793]],
                [[-0.89111614, 1.2611268], [-1.0644908, 0.69448]],
                [[0.19064833, 0.80228466], [0.6924452, -1.6853783]],
            ]
        );

        let g = y1.exp().mean().backward();
        assert_close_to_literal!(bn.running_mean, [-0.0175438, -0.0214163, 0.0268384]);
        assert_close_to_literal!(bn.running_var, [1.1361228, 1.0889612, 1.3478994]);
        assert_close_to_literal!(g.get(&bn.scale), [0.2506705, 0.4257624, 0.257648]);
        assert_close_to_literal!(g.get(&bn.bias), [0.4663894, 0.5239304, 0.4687197]);
        assert_close_to_literal!(
            g.get(&x1),
            [
                [[0.0030178577, 0.011973545], [0.0038383976, -0.018829815]],
                [[-0.0016367957, 0.024275035], [0.0092941, -0.03193234]],
                [[-0.015617318, 0.009291172], [0.0026013851, 0.0037247613]],
            ]
        );
    }

    #[test]
    fn test_batchnorm2d_4d_forward_mut() {
        let dev = TestDevice::seed_from_u64(2);

        let x1: Tensor<Rank4<2, 2, 2, 3>, TestDtype, _> = dev.sample_normal();
        let mut bn = dev.build_module::<TestDtype>(BatchNorm2DConstConfig::<2>::default());

        let y1 = bn.forward_mut(x1.leaky_trace());
        #[rustfmt::skip]
        assert_close_to_literal!(
            y1,
            [
                [[[-0.93348885, -2.1979978, 0.19754872],[0.29159376, -0.6282544, -1.0415624]], [[1.1156346, 0.89029306, -1.1608727],[-0.73874927, 0.13254784, -0.77676374]]],
                [[[0.60655713, 0.62703574, 0.12648833],[1.5577206, 0.18830705, 1.2060523]],[[0.37415895, -0.9069047, -0.9519587],[-0.02608296, 2.3435123, -0.2948149]]],
            ]
        );

        let g = y1.exp().mean().backward();
        assert_close_to_literal!(bn.running_mean, [-0.02424082, 0.00407672]);
        assert_close_to_literal!(bn.running_var, [0.9676103, 1.0458221]);
        assert_close_to_literal!(g.get(&bn.scale), [0.5582906, 1.1929206]);
        assert_close_to_literal!(g.get(&bn.bias), [0.7535024, 0.92750454]);
        #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&x1),
            [
                [[[-0.00378475, 0.05601016, -0.02694868],[-0.02614748, -0.01439525, 0.00047035]],[[-0.05280511, -0.05561727, 0.04425058],[0.01388359, -0.03710236, 0.01651]]],
                [[[-0.01853323, -0.01773504, -0.02717264],[0.0794776, -0.02699574, 0.02575465]],[[-0.04663141, 0.02567738, 0.0289102],[-0.0294986, 0.10708933, -0.01466625]]],
            ]
        );
    }

    #[test]
    fn test_batchnorm2d_3d_repeated_forward_mut() {
        let dev = TestDevice::seed_from_u64(12);

        let x1: Tensor<Rank3<3, 4, 5>, TestDtype, _> = dev.sample_normal();
        let mut bn = dev.build_module::<TestDtype>(BatchNorm2DConstConfig::<3>::default());

        let _ = bn.forward_mut(x1.leaky_trace());
        assert_close_to_literal!(bn.running_mean, [0.0083191, -0.0370511, -0.0079481]);
        assert_close_to_literal!(bn.running_var, [1.0344709, 0.9340682, 1.0266376]);

        let _ = bn.forward_mut(x1.leaky_trace());
        assert_close_to_literal!(bn.running_mean, [0.0158063, -0.0703971, -0.0151013]);
        assert_close_to_literal!(bn.running_var, [1.0654946, 0.87472963, 1.0506116]);

        let _ = bn.forward_mut(x1.leaky_trace());
        assert_close_to_literal!(bn.running_mean, [0.0225448, -0.1004085, -0.0215393]);
        assert_close_to_literal!(bn.running_var, [1.093416, 0.8213248, 1.0721881]);

        let _ = bn.forward_mut(x1.leaky_trace());
        assert_close_to_literal!(bn.running_mean, [0.0286095, -0.1274188, -0.0273335]);
        assert_close_to_literal!(bn.running_var, [1.1185452, 0.7732605, 1.0916069]);

        let m = bn.running_mean.clone();
        let v = bn.running_var.clone();

        let x2 = dev.sample_normal::<Rank3<3, 2, 2>>();
        let y2 = bn.forward(x2);
        // running stats shouldn't have been updated
        assert_eq!(bn.running_mean.array(), m.array());
        assert_eq!(bn.running_var.array(), v.array());
        assert_close_to_literal!(
            y2,
            [
                [[0.0897828, -0.01880704], [-0.55082226, -0.50515544]],
                [[0.13778551, 0.25317147], [-1.2689502, 0.61595416]],
                [[0.73018146, 0.3243845], [-1.1041277, 0.38778353]],
            ]
        );
    }

    #[test]
    fn test_batchnorm2d_update() {
        let dev: TestDevice = Default::default();

        let x1: Tensor<Rank3<3, 4, 5>, TestDtype, _> = dev.sample_normal();
        let mut bn = dev.build_module::<TestDtype>(BatchNorm2DConstConfig::default());
        let y = bn.forward_mut(x1.leaky_trace());
        let g = y.square().mean().backward();

        let mut opt = crate::nn::optim::Sgd::new(&bn, Default::default());
        opt.update(&mut bn, &g).expect("");
    }

    #[derive(Default, Clone, Sequential)]
    struct Arch {
        pub batch: BatchNorm2DConstConfig<3>,
    }

    #[test]
    fn test_batchnorm2d_update_with_derive() {
        let dev: TestDevice = Default::default();

        let x1: Tensor<Rank3<3, 4, 5>, TestDtype, _> = dev.sample_normal();
        let mut bn = dev.build_module::<TestDtype>(Arch::default());
        let y = bn.forward_mut(x1.leaky_trace());
        let g = y.square().mean().backward();

        let mut opt = crate::nn::optim::Sgd::new(&bn, Default::default());
        opt.update(&mut bn, &g).expect("");
    }
}
