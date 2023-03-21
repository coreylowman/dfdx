use crate::{shapes::*, tensor::*, tensor_ops::*};
use num_traits::FromPrimitive;

use super::{
    batchnorm2d::{infer_fwd, train_fwd},
    *,
};

pub mod builder {
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub struct BatchNorm1D<const C: usize>;
}

impl<const C: usize, E: Dtype, D: Device<E>> BuildOnDevice<D, E> for builder::BatchNorm1D<C>
where
    BatchNorm1D<C, E, D>: BuildModule<D, E>,
{
    type Built = BatchNorm1D<C, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, D::Err> {
        Self::Built::try_build(device)
    }
}

/// Batch normalization for sequences as described in
/// [Batch Normalization: Accelerating Deep Network Training
/// by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
///
/// Generics:
///
/// - `C` the size of the dimension to reduce. Both for 2d tensors (of the form <BATCH_SIZE, DIMENSION>)
///   as well as 3d tensors (of the form <BATCH_SIZE, DIMENSION, SEQUENCE_LENGTH>), this is the 1st dimension.
///
/// # Training vs Inference
///
/// BatchNorm1D supports the following cases (see sections below for more details):
/// 1. **Training**: [ModuleMut] and [OwnedTape] on the input tensor
/// 2. **Inference**: [Module] and [NoneTape] on the input tensor.
///
/// *NOTE: ModuleMut/NoneTape, and Module/OwnedTape will fail to compile.*
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// type Model = BatchNorm1D<3>;
/// let bn = dev.build_module::<Model, f32>();
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
#[derive(Clone, Debug)]
pub struct BatchNorm1D<const C: usize, E: Dtype, D: DeviceStorage> {
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

impl<const C: usize, E: Dtype, D: Device<E>> BatchNorm1D<C, E, D> {
    /// generic forward for inference
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

impl<B: Dim, const C: usize, E: Dtype, D: Device<E>> Module<Tensor<(B, Const<C>), E, D, NoneTape>>
    for BatchNorm1D<C, E, D>
{
    type Output = Tensor<(B, Const<C>), E, D, NoneTape>;
    type Error = D::Err;

    /// Inference 1d forward - does **not** update [Self::running_mean] and [Self::running_var]
    fn try_forward(
        &self,
        x: Tensor<(B, Const<C>), E, D, NoneTape>,
    ) -> Result<Self::Output, D::Err> {
        self.infer_fwd(x)
    }
}

impl<B: Dim, const C: usize, L: Dim, E: Dtype, D: Device<E>>
    Module<Tensor<(B, Const<C>, L), E, D, NoneTape>> for BatchNorm1D<C, E, D>
{
    type Output = Tensor<(B, Const<C>, L), E, D, NoneTape>;
    type Error = D::Err;

    /// Inference 2d forward - does **not** update [Self::running_mean] and [Self::running_var]
    fn try_forward(
        &self,
        x: Tensor<(B, Const<C>, L), E, D, NoneTape>,
    ) -> Result<Self::Output, D::Err> {
        self.infer_fwd(x)
    }
}

impl<B: Dim, const C: usize, L: Dim, E: Dtype, D: Device<E>>
    ModuleMut<Tensor<(B, Const<C>, L), E, D, OwnedTape<E, D>>> for BatchNorm1D<C, E, D>
{
    type Output = Tensor<(B, Const<C>, L), E, D, OwnedTape<E, D>>;
    type Error = D::Err;

    /// Training 1d forward - updates [Self::running_mean] and [Self::running_var]
    fn try_forward_mut(
        &mut self,
        x: Tensor<(B, Const<C>, L), E, D, OwnedTape<E, D>>,
    ) -> Result<Self::Output, D::Err> {
        self.train_fwd(x)
    }
}

impl<B: Dim, const C: usize, E: Dtype, D: Device<E>>
    ModuleMut<Tensor<(B, Const<C>), E, D, OwnedTape<E, D>>> for BatchNorm1D<C, E, D>
{
    type Output = Tensor<(B, Const<C>), E, D, OwnedTape<E, D>>;
    type Error = D::Err;

    /// Training 2d forward - updates [Self::running_mean] and [Self::running_var]
    fn try_forward_mut(
        &mut self,
        x: Tensor<(B, Const<C>), E, D, OwnedTape<E, D>>,
    ) -> Result<Self::Output, D::Err> {
        self.train_fwd(x)
    }
}

impl<const C: usize, E: Dtype, D: Device<E>> TensorCollection<E, D> for BatchNorm1D<C, E, D> {
    type To<E2: Dtype, D2: Device<E2>> = BatchNorm1D<C, E2, D2>;

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
            |(scale, bias, running_mean, running_var)| BatchNorm1D {
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
    use super::builder::BatchNorm1D;
    use crate::{nn::builders::*, optim::*, shapes::*, tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_batchnorm1d_2d_forward_mut() {
        let dev = TestDevice::seed_from_u64(0);

        let x1: Tensor<Rank2<3, 2>, TestDtype, _> = dev.sample(rand_distr::StandardNormal);
        let mut bn = BatchNorm1D::<2>::build_on_device(&dev);

        let y1 = bn.forward_mut(x1.leaky_trace());
        assert_close(
            &y1.array(),
            &[
                [1.3168651, 0.19157785],
                [-1.1049646, -1.3092154],
                [-0.21190044, 1.1176374],
            ],
        );

        let g = y1.exp().mean().backward();
        assert_close(&bn.running_mean.array(), &[-0.09994803, 0.07696156]);
        assert_close(&bn.running_var.array(), &[1.1536077, 0.9321649]);
        assert_close(&g.get(&bn.scale).array(), &[0.72945416, 0.5493023]);
        assert_close(&g.get(&bn.bias).array(), &[0.8119954, 0.7564688]);
        assert_close(
            &g.get(&x1).array(),
            &[
                [0.023908734, -0.18436226],
                [0.040923715, 0.0703277],
                [-0.06483248, 0.11403453],
            ],
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
        let mut bn = BatchNorm1D::<DIMENSION>::build_on_device(&dev);

        let y1 = bn.forward_mut(x1.leaky_trace());
        assert_close(
            &y1.array(),
            &[
                [[0.059494145, 0.21366562], [-1.0539212, 0.5588659]],
                [[-2.0465322, 0.6680055], [-0.46153978, 0.8375814]],
                [[-0.041158404, 1.1465254], [1.411404, -1.2923905]],
            ],
        );

        let g = y1.exp().mean().backward();
        assert_close(&bn.running_mean.array(), &[0.065665804, -0.07374697]);
        assert_close(&bn.running_var.array(), &[1.0069065, 1.2117702]);
        assert_close(&g.get(&bn.scale).array(), &[0.4112549, 0.6407272]);
        assert_close(&g.get(&bn.bias).array(), &[0.7071625, 0.78455544]);
        assert_close(
            &g.get(&x1).array(),
            &[
                [[-0.035488494, -0.031065114], [0.0067214966, -0.02774144]],
                [[0.035152107, -0.0011850521], [-0.017958358, -0.017146945]],
                [[-0.03715139, 0.0697379], [0.037428252, 0.018696927]],
            ],
        );
    }

    #[test]
    fn test_batchnorm1d_update() {
        const BATCH_SIZE: usize = 3;
        const DIMENSION: usize = 4;
        let dev: TestDevice = Default::default();

        let x1: Tensor<Rank2<BATCH_SIZE, DIMENSION>, TestDtype, _> = dev.sample_normal();
        let mut bn = dev.build_module::<BatchNorm1D<DIMENSION>, TestDtype>();
        let y = bn.forward_mut(x1.leaky_trace());
        let g = y.square().mean().backward();

        let mut opt = Sgd::new(&bn, Default::default());
        opt.update(&mut bn, &g).expect("");
    }
}
