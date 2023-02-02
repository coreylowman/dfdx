use crate::{gradients::*, optim::*, shapes::*, tensor::*, tensor_ops::*};

use super::{BuildModule, BuildOnDevice, Module, ModuleMut, ResetParams, ToDevice};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct BatchNorm2D<const C: usize>;
impl<const C: usize, E: Dtype, D: DeviceStorage> BuildOnDevice<D, E> for BatchNorm2D<C>
where
    DeviceBatchNorm2D<C, D>: BuildModule<D, E>,
{
    type Built = DeviceBatchNorm2D<C, D>;
}

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
/// # let dev: Cpu = Default::default();
/// type Model = BatchNorm2D<3>;
/// let bn = Model::build_on_device(&dev);
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
#[derive(Clone, Debug)]
pub struct DeviceBatchNorm2D<const C: usize, D: DeviceStorage> {
    /// Scale for affine transform. Defaults to 1.0
    pub scale: Tensor<Rank1<C>, f32, D>,
    /// Bias for affine transform. Defaults to 0.0
    pub bias: Tensor<Rank1<C>, f32, D>,
    /// Spatial mean that is updated during training. Defaults to 0.0
    pub running_mean: Tensor<Rank1<C>, f32, D>,
    /// Spatial variance that is updated during training. Defaults to 1.0
    pub running_var: Tensor<Rank1<C>, f32, D>,
    /// Added to variance before taking sqrt for numerical stability. Defaults to 1e-5
    pub epsilon: f32,
    /// Controls exponential moving average of running stats.Defaults to 0.1
    ///
    /// `running_stat * (1.0 - momentum) + stat * momentum`.
    pub momentum: f32,
}

impl<const C: usize, D: Device<f32>> DeviceBatchNorm2D<C, D> {
    /// generic forward for inference
    fn infer_fwd<S: Shape, Ax: Axes>(&self, x: Tensor<S, f32, D>) -> Tensor<S, f32, D>
    where
        Rank1<C>: BroadcastShapeTo<S, Ax>,
    {
        let shape = *x.shape();

        // statistics for normalizing
        let std = (self.running_var.clone() + self.epsilon).sqrt();
        let mean = self.running_mean.clone();

        // normalize & affine
        let x = sub(x, mean.broadcast_like(&shape));
        let x = div(x, std.broadcast_like(&shape));
        let x = mul(x, self.scale.clone().broadcast_like(&shape));
        add(x, self.bias.clone().broadcast_like(&shape))
    }

    fn train_fwd<S: Shape, T: Tape<D>, Ax: Axes>(
        &mut self,
        x: Tensor<S, f32, D, T>,
    ) -> Tensor<S, f32, D, T>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Rank1<C>, Ax>,
    {
        let n = <S as HasAxes<Ax>>::size(x.shape()) as f32;
        let shape = *x.shape();

        // compute statistics for updating running stats later - on tape
        let mean_chan = x.retaped::<T>().mean::<Rank1<C>, _>();

        // update statistics since we are training - off tape
        self.running_mean = self.running_mean.clone() * (1.0 - self.momentum)
            + mean_chan.retaped::<NoneTape>() * self.momentum;

        let mean = mean_chan.broadcast_like(&shape);
        let centered = x - mean;

        let var_chan = centered.retaped::<T>().square().mean::<Rank1<C>, _>();

        // NOTE: uses unbiased variance in running estimate
        self.running_var = self.running_var.clone() * (1.0 - self.momentum)
            + var_chan.retaped::<NoneTape>() * (self.momentum * n / (n - 1.0));

        // statistics for normalizing - on tape
        let std = (var_chan + self.epsilon).sqrt().broadcast_like(&shape);

        // record broadcast of scale & bias - on tape
        let scale = self.scale.retaped::<T>().broadcast_like(&shape);
        let bias = self.bias.retaped::<T>().broadcast_like(&shape);

        // normalize & affine - on tape
        (centered / std) * scale + bias
    }
}

impl<const C: usize, H: Dim, W: Dim, D: Device<f32>>
    Module<Tensor<(Const<C>, H, W), f32, D, NoneTape>> for DeviceBatchNorm2D<C, D>
{
    type Output = Tensor<(Const<C>, H, W), f32, D, NoneTape>;

    /// Inference 3d forward - does **not** update [Self::running_mean] and [Self::running_var]
    fn forward(&self, x: Tensor<(Const<C>, H, W), f32, D, NoneTape>) -> Self::Output {
        self.infer_fwd(x)
    }
}

impl<B: Dim, const C: usize, H: Dim, W: Dim, D: Device<f32>>
    Module<Tensor<(B, Const<C>, H, W), f32, D, NoneTape>> for DeviceBatchNorm2D<C, D>
{
    type Output = Tensor<(B, Const<C>, H, W), f32, D, NoneTape>;

    /// Inference 4d forward - does **not** update [Self::running_mean] and [Self::running_var]
    fn forward(&self, x: Tensor<(B, Const<C>, H, W), f32, D, NoneTape>) -> Self::Output {
        self.infer_fwd(x)
    }
}

impl<const C: usize, H: Dim, W: Dim, D: Device<f32>>
    ModuleMut<Tensor<(Const<C>, H, W), f32, D, OwnedTape<D>>> for DeviceBatchNorm2D<C, D>
{
    type Output = Tensor<(Const<C>, H, W), f32, D, OwnedTape<D>>;

    /// Training 3d forward - updates [Self::running_mean] and [Self::running_var]
    fn forward_mut(&mut self, x: Tensor<(Const<C>, H, W), f32, D, OwnedTape<D>>) -> Self::Output {
        self.train_fwd(x)
    }
}

impl<B: Dim, const C: usize, H: Dim, W: Dim, D: Device<f32>>
    ModuleMut<Tensor<(B, Const<C>, H, W), f32, D, OwnedTape<D>>> for DeviceBatchNorm2D<C, D>
{
    type Output = Tensor<(B, Const<C>, H, W), f32, D, OwnedTape<D>>;

    /// Training 4d forward - updates [Self::running_mean] and [Self::running_var]
    fn forward_mut(
        &mut self,
        x: Tensor<(B, Const<C>, H, W), f32, D, OwnedTape<D>>,
    ) -> Self::Output {
        self.train_fwd(x)
    }
}

impl<const C: usize, D: Device<f32>> BuildModule<D, f32> for DeviceBatchNorm2D<C, D> {
    fn try_build(device: &D) -> Result<Self, D::Err> {
        Ok(Self {
            scale: device.try_ones()?,
            bias: device.try_zeros()?,
            running_mean: device.try_zeros()?,
            running_var: device.try_ones()?,
            epsilon: 1e-5,
            momentum: 0.1,
        })
    }
}

impl<const C: usize, D: Device<f32>> ResetParams<D, f32> for DeviceBatchNorm2D<C, D> {
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        self.scale.try_fill_with_ones()?;
        self.bias.try_fill_with_zeros()?;
        self.running_mean.try_fill_with_zeros()?;
        self.running_var.try_fill_with_ones()?;
        Ok(())
    }
}

impl<const C: usize, D1: Device<f32>, D2: Device<f32>> ToDevice<D2> for DeviceBatchNorm2D<C, D1> {
    type Output = DeviceBatchNorm2D<C, D2>;
    fn to_device(&self, device: &D2) -> Self::Output {
        DeviceBatchNorm2D {
            scale: self.scale.to_device(device),
            bias: self.bias.to_device(device),
            running_mean: self.running_mean.to_device(device),
            running_var: self.running_var.to_device(device),
            epsilon: self.epsilon,
            momentum: self.momentum,
        }
    }
}

impl<const C: usize, D: Device<f32>> GradientUpdate<D, f32> for DeviceBatchNorm2D<C, D> {
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), <D>::Err>
    where
        U: ParamUpdater<D, f32>,
    {
        self.scale.update(updater, unused)?;
        self.bias.update(updater, unused)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_batchnorm2d_3d_forward_mut() {
        let dev = TestDevice::seed_from_u64(0);

        let x1: Tensor<Rank3<3, 2, 2>, f32, _> = dev.sample(rand_distr::StandardNormal);
        let mut bn: DeviceBatchNorm2D<3, _> = BuildModule::build(&dev);

        let y1 = bn.forward_mut(x1.trace());
        assert_close(
            &y1.array(),
            &[
                [[0.66747534, 0.77682495], [-1.698878, 0.25457793]],
                [[-0.89111614, 1.2611268], [-1.0644908, 0.69448]],
                [[0.19064833, 0.80228466], [0.6924452, -1.6853783]],
            ],
        );

        let g = y1.exp().mean().backward();
        assert_close(
            &bn.running_mean.array(),
            &[-0.0175438, -0.0214163, 0.0268384],
        );
        assert_close(&bn.running_var.array(), &[1.1361228, 1.0889612, 1.3478994]);
        assert_close(&g.get(&bn.scale).array(), &[0.2506705, 0.4257624, 0.257648]);
        assert_close(&g.get(&bn.bias).array(), &[0.4663894, 0.5239304, 0.4687197]);
        assert_close(
            &g.get(&x1).array(),
            &[
                [[0.0030178577, 0.011973545], [0.0038383976, -0.018829815]],
                [[-0.0016367957, 0.024275035], [0.0092941, -0.03193234]],
                [[-0.015617318, 0.009291172], [0.0026013851, 0.0037247613]],
            ],
        );
    }

    #[test]
    fn test_batchnorm2d_4d_forward_mut() {
        let dev = TestDevice::seed_from_u64(2);

        let x1 = dev.sample_normal::<Rank4<2, 2, 2, 3>>();
        let mut bn = BatchNorm2D::<2>::build_on_device(&dev);

        let y1 = bn.forward_mut(x1.trace());
        #[rustfmt::skip]
        assert_close(
            &y1.array(),
            &[
                [[[-0.93348885, -2.1979978, 0.19754872],[0.29159376, -0.6282544, -1.0415624]], [[1.1156346, 0.89029306, -1.1608727],[-0.73874927, 0.13254784, -0.77676374]]],
                [[[0.60655713, 0.62703574, 0.12648833],[1.5577206, 0.18830705, 1.2060523]],[[0.37415895, -0.9069047, -0.9519587],[-0.02608296, 2.3435123, -0.2948149]]],
            ],
        );

        let g = y1.exp().mean().backward();
        assert_close(&bn.running_mean.array(), &[-0.02424082, 0.00407672]);
        assert_close(&bn.running_var.array(), &[0.9676103, 1.0458221]);
        assert_close(&g.get(&bn.scale).array(), &[0.5582906, 1.1929206]);
        assert_close(&g.get(&bn.bias).array(), &[0.7535024, 0.92750454]);
        #[rustfmt::skip]
        assert_close(
            &g.get(&x1).array(),
            &[
                [[[-0.00378475, 0.05601016, -0.02694868],[-0.02614748, -0.01439525, 0.00047035]],[[-0.05280511, -0.05561727, 0.04425058],[0.01388359, -0.03710236, 0.01651]]],
                [[[-0.01853323, -0.01773504, -0.02717264],[0.0794776, -0.02699574, 0.02575465]],[[-0.04663141, 0.02567738, 0.0289102],[-0.0294986, 0.10708933, -0.01466625]]],
            ],
        );
    }

    #[test]
    fn test_batchform2d_3d_repeated_forward_mut() {
        let dev = TestDevice::seed_from_u64(12);

        let x1 = dev.sample_normal::<Rank3<3, 4, 5>>();
        let mut bn: DeviceBatchNorm2D<3, _> = BuildModule::build(&dev);

        let _ = bn.forward_mut(x1.trace());
        assert_close(
            &bn.running_mean.array(),
            &[0.0083191, -0.0370511, -0.0079481],
        );
        assert_close(&bn.running_var.array(), &[1.0344709, 0.9340682, 1.0266376]);

        let _ = bn.forward_mut(x1.trace());
        assert_close(
            &bn.running_mean.array(),
            &[0.0158063, -0.0703971, -0.0151013],
        );
        assert_close(&bn.running_var.array(), &[1.0654946, 0.87472963, 1.0506116]);

        let _ = bn.forward_mut(x1.trace());
        assert_close(
            &bn.running_mean.array(),
            &[0.0225448, -0.1004085, -0.0215393],
        );
        assert_close(&bn.running_var.array(), &[1.093416, 0.8213248, 1.0721881]);

        let _ = bn.forward_mut(x1.trace());
        assert_close(
            &bn.running_mean.array(),
            &[0.0286095, -0.1274188, -0.0273335],
        );
        assert_close(&bn.running_var.array(), &[1.1185452, 0.7732605, 1.0916069]);

        let m = bn.running_mean.clone();
        let v = bn.running_var.clone();

        let x2 = dev.sample_normal::<Rank3<3, 2, 2>>();
        let y2 = bn.forward(x2);
        // running stats shouldn't have been updated
        assert_eq!(bn.running_mean.array(), m.array());
        assert_eq!(bn.running_var.array(), v.array());
        assert_close(
            &y2.array(),
            &[
                [[0.0897828, -0.01880704], [-0.55082226, -0.50515544]],
                [[0.13778551, 0.25317147], [-1.2689502, 0.61595416]],
                [[0.73018146, 0.3243845], [-1.1041277, 0.38778353]],
            ],
        );
    }
}
