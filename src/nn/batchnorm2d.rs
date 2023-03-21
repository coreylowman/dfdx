use crate::{shapes::*, tensor::*, tensor_ops::*};
use num_traits::FromPrimitive;

use super::*;

pub mod builder {
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub struct BatchNorm2D<const C: usize>;
}

impl<const C: usize, E: Dtype, D: Device<E>> BuildOnDevice<D, E> for builder::BatchNorm2D<C>
where
    BatchNorm2D<C, E, D>: BuildModule<D, E>,
{
    type Built = BatchNorm2D<C, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, D::Err> {
        Self::Built::try_build(device)
    }
}

/// generic batchnorm forward for training
pub fn train_fwd<const C: usize, S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>, Ax: Axes>(
    x: Tensor<S, E, D, T>,
    var: &mut Tensor<Rank1<C>, E, D>,
    mean: &mut Tensor<Rank1<C>, E, D>,
    scale: &Tensor<Rank1<C>, E, D>,
    bias: &Tensor<Rank1<C>, E, D>,
    epsilon: E,
    momentum: E,
) -> Result<Tensor<S, E, D, T>, D::Err>
where
    S: HasAxes<Ax> + ReduceShapeTo<Rank1<C>, Ax>,
{
    let n = E::from_usize(<S as HasAxes<Ax>>::size(x.shape())).unwrap();
    let shape = *x.shape();

    // compute statistics for updating running stats later - on tape
    let mean_chan = x.retaped::<T>().try_mean::<Rank1<C>, _>()?;

    // update statistics since we are training - off tape
    mean.try_axpy(E::ONE - momentum, &mean_chan, momentum)?;

    let centered = x.try_sub(mean_chan.try_broadcast_like(&shape)?)?;

    let var_chan = centered.retaped::<T>().square().mean::<Rank1<C>, _>();

    // NOTE: uses unbiased variance in running estimate
    var.try_axpy(E::ONE - momentum, &var_chan, momentum * n / (n - E::ONE))?;

    // statistics for normalizing - on tape
    let std = var_chan.try_add(epsilon)?.try_sqrt()?;

    // record broadcast of scale & bias - on tape
    let scale = scale
        .retaped::<T>()
        .try_div(std)?
        .try_broadcast_like(&shape)?;
    let bias = bias.retaped::<T>().try_broadcast_like(&shape)?;

    // normalize & affine - on tape
    centered.try_mul(scale)?.try_add(bias)
}

/// generic batchnorm forward for inference
pub fn infer_fwd<const C: usize, S: Shape, E: Dtype, D: Device<E>, Ax: Axes>(
    x: Tensor<S, E, D>,
    var: &Tensor<Rank1<C>, E, D>,
    mean: &Tensor<Rank1<C>, E, D>,
    scale: &Tensor<Rank1<C>, E, D>,
    bias: &Tensor<Rank1<C>, E, D>,
    epsilon: E,
) -> Result<Tensor<S, E, D>, D::Err>
where
    Rank1<C>: BroadcastShapeTo<S, Ax>,
{
    let shape = *x.shape();

    // statistics for normalizing
    let std = (var.clone() + epsilon).try_sqrt()?;
    let mean = mean.clone();

    // normalize & affine
    let x = x.try_sub(mean.try_broadcast_like(&shape)?)?;
    let x = x.try_div(std.try_broadcast_like(&shape)?)?;
    let x = x.try_mul(scale.clone().try_broadcast_like(&shape)?)?;
    x.try_add(bias.clone().try_broadcast_like(&shape)?)
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
/// let bn = dev.build_module::<Model, f32>();
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
pub struct BatchNorm2D<const C: usize, E: Dtype, D: DeviceStorage> {
    /// Scale for affine transform. Defaults to 1.0
    pub scale: Tensor<Rank1<C>, E, D>,
    /// Bias for affine transform. Defaults to 0.0
    pub bias: Tensor<Rank1<C>, E, D>,
    /// Spatial mean that is updated during training. Defaults to 0.0
    pub running_mean: Tensor<Rank1<C>, E, D>,
    /// Spatial variance that is updated during training. Defaults to 1.0
    pub running_var: Tensor<Rank1<C>, E, D>,
    /// Added to variance before taking sqrt for numerical stability. Defaults to 1e-5
    pub epsilon: E,
    /// Controls exponential moving average of running stats.Defaults to 0.1
    ///
    /// `running_stat * (1.0 - momentum) + stat * momentum`.
    pub momentum: E,
}

impl<const C: usize, E: Dtype, D: Device<E>> BatchNorm2D<C, E, D> {
    fn infer_fwd<S: Shape, Ax: Axes>(&self, x: Tensor<S, E, D>) -> Result<Tensor<S, E, D>, D::Err>
    where
        Rank1<C>: BroadcastShapeTo<S, Ax>,
    {
        infer_fwd(
            x,
            &self.running_var,
            &self.running_mean,
            &self.scale,
            &self.bias,
            self.epsilon,
        )
    }

    fn train_fwd<S: Shape, T: Tape<E, D>, Ax: Axes>(
        &mut self,
        x: Tensor<S, E, D, T>,
    ) -> Result<Tensor<S, E, D, T>, D::Err>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Rank1<C>, Ax>,
    {
        train_fwd(
            x,
            &mut self.running_var,
            &mut self.running_mean,
            &self.scale,
            &self.bias,
            self.epsilon,
            self.momentum,
        )
    }
}

impl<const C: usize, H: Dim, W: Dim, E: Dtype, D: Device<E>>
    Module<Tensor<(Const<C>, H, W), E, D, NoneTape>> for BatchNorm2D<C, E, D>
{
    type Output = Tensor<(Const<C>, H, W), E, D, NoneTape>;
    type Error = D::Err;

    /// Inference 3d forward - does **not** update [Self::running_mean] and [Self::running_var]
    fn try_forward(
        &self,
        x: Tensor<(Const<C>, H, W), E, D, NoneTape>,
    ) -> Result<Self::Output, D::Err> {
        self.infer_fwd(x)
    }
}

impl<B: Dim, const C: usize, H: Dim, W: Dim, E: Dtype, D: Device<E>>
    Module<Tensor<(B, Const<C>, H, W), E, D, NoneTape>> for BatchNorm2D<C, E, D>
{
    type Output = Tensor<(B, Const<C>, H, W), E, D, NoneTape>;
    type Error = D::Err;

    /// Inference 4d forward - does **not** update [Self::running_mean] and [Self::running_var]
    fn try_forward(
        &self,
        x: Tensor<(B, Const<C>, H, W), E, D, NoneTape>,
    ) -> Result<Self::Output, D::Err> {
        self.infer_fwd(x)
    }
}

impl<const C: usize, H: Dim, W: Dim, E: Dtype, D: Device<E>>
    ModuleMut<Tensor<(Const<C>, H, W), E, D, OwnedTape<E, D>>> for BatchNorm2D<C, E, D>
{
    type Output = Tensor<(Const<C>, H, W), E, D, OwnedTape<E, D>>;
    type Error = D::Err;

    /// Training 3d forward - updates [Self::running_mean] and [Self::running_var]
    fn try_forward_mut(
        &mut self,
        x: Tensor<(Const<C>, H, W), E, D, OwnedTape<E, D>>,
    ) -> Result<Self::Output, D::Err> {
        self.train_fwd(x)
    }
}

impl<B: Dim, const C: usize, H: Dim, W: Dim, E: Dtype, D: Device<E>>
    ModuleMut<Tensor<(B, Const<C>, H, W), E, D, OwnedTape<E, D>>> for BatchNorm2D<C, E, D>
{
    type Output = Tensor<(B, Const<C>, H, W), E, D, OwnedTape<E, D>>;
    type Error = D::Err;

    /// Training 4d forward - updates [Self::running_mean] and [Self::running_var]
    fn try_forward_mut(
        &mut self,
        x: Tensor<(B, Const<C>, H, W), E, D, OwnedTape<E, D>>,
    ) -> Result<Self::Output, D::Err> {
        self.train_fwd(x)
    }
}

impl<const C: usize, E: Dtype, D: Device<E>> TensorCollection<E, D> for BatchNorm2D<C, E, D> {
    type To<E2: Dtype, D2: Device<E2>> = BatchNorm2D<C, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::tensor(
                    "scale",
                    |s| &s.scale,
                    |s| &mut s.scale,
                    TensorOptions::reset_to_ones(),
                ),
                Self::tensor(
                    "bias",
                    |s| &s.bias,
                    |s| &mut s.bias,
                    TensorOptions::reset_to_zeros(),
                ),
                Self::tensor(
                    "running_mean",
                    |s| &s.running_mean,
                    |s| &mut s.running_mean,
                    TensorOptions::detached(|t| t.try_fill_with_zeros()),
                ),
                Self::tensor(
                    "running_var",
                    |s| &s.running_var,
                    |s| &mut s.running_var,
                    TensorOptions::detached(|t| t.try_fill_with_ones()),
                ),
            ),
            |(scale, bias, running_mean, running_var)| BatchNorm2D {
                scale,
                bias,
                running_mean,
                running_var,
                epsilon: V::E2::from_f32(1e-5).unwrap(),
                momentum: V::E2::from_f32(0.1).unwrap(),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::builder::BatchNorm2D;
    use crate::{nn::builders::*, optim::*, shapes::*, tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_batchnorm2d_3d_forward_mut() {
        let dev = TestDevice::seed_from_u64(0);

        let x1: Tensor<Rank3<3, 2, 2>, TestDtype, _> = dev.sample(rand_distr::StandardNormal);
        let mut bn = BatchNorm2D::<3>::build_on_device(&dev);

        let y1 = bn.forward_mut(x1.leaky_trace());
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

        let x1: Tensor<Rank4<2, 2, 2, 3>, TestDtype, _> = dev.sample_normal();
        let mut bn = BatchNorm2D::<2>::build_on_device(&dev);

        let y1 = bn.forward_mut(x1.leaky_trace());
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
    fn test_batchnorm2d_3d_repeated_forward_mut() {
        let dev = TestDevice::seed_from_u64(12);

        let x1: Tensor<Rank3<3, 4, 5>, TestDtype, _> = dev.sample_normal();
        let mut bn = BatchNorm2D::<3>::build_on_device(&dev);

        let _ = bn.forward_mut(x1.leaky_trace());
        assert_close(
            &bn.running_mean.array(),
            &[0.0083191, -0.0370511, -0.0079481],
        );
        assert_close(&bn.running_var.array(), &[1.0344709, 0.9340682, 1.0266376]);

        let _ = bn.forward_mut(x1.leaky_trace());
        assert_close(
            &bn.running_mean.array(),
            &[0.0158063, -0.0703971, -0.0151013],
        );
        assert_close(&bn.running_var.array(), &[1.0654946, 0.87472963, 1.0506116]);

        let _ = bn.forward_mut(x1.leaky_trace());
        assert_close(
            &bn.running_mean.array(),
            &[0.0225448, -0.1004085, -0.0215393],
        );
        assert_close(&bn.running_var.array(), &[1.093416, 0.8213248, 1.0721881]);

        let _ = bn.forward_mut(x1.leaky_trace());
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

    #[test]
    fn test_batchnorm2d_update() {
        let dev: TestDevice = Default::default();

        let x1: Tensor<Rank3<3, 4, 5>, TestDtype, _> = dev.sample_normal();
        let mut bn = dev.build_module::<BatchNorm2D<3>, TestDtype>();
        let y = bn.forward_mut(x1.leaky_trace());
        let g = y.square().mean().backward();

        let mut opt = Sgd::new(&bn, Default::default());
        opt.update(&mut bn, &g).expect("");
    }
}
