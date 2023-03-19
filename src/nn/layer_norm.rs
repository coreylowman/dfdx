use crate::{shapes::*, tensor::*, tensor_ops::*};
use num_traits::FromPrimitive;

use super::*;

pub mod builder {
    #[derive(Debug)]
    pub struct LayerNorm1D<const M: usize>;
}
impl<const M: usize, E: Dtype, D: Device<E>> BuildOnDevice<D, E> for builder::LayerNorm1D<M>
where
    LayerNorm1D<M, E, D>: BuildModule<D, E>,
{
    type Built = LayerNorm1D<M, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, D::Err> {
        Self::Built::try_build(device)
    }
}

/// Implements layer normalization as described in [Layer Normalization](https://arxiv.org/abs/1607.06450).
///
/// This calls [normalize()] on the last axis of the input to normalize to 0 mean and unit std dev, and then does an element-wise
/// affine transform using learnable parameters [Self::gamma] and [Self::beta].
///
/// [Self::epsilon] is passed to [normalize()] and added to the variance to ensure big enough numbers. It defaults to `1e-5`.
///
/// # Generics
/// - `M` The size of the affine transform tensors.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// type Model = LayerNorm1D<5>;
/// let model = dev.build_module::<Model, f32>();
/// let _: Tensor<Rank1<5>, f32, _> = model.forward(dev.zeros::<Rank1<5>>());
/// ```

#[derive(Debug, Clone)]
pub struct LayerNorm1D<const M: usize, E: Dtype, D: DeviceStorage> {
    pub gamma: Tensor<Rank1<M>, E, D>,
    pub beta: Tensor<Rank1<M>, E, D>,
    pub epsilon: E,
}

impl<const M: usize, E: Dtype, D: DeviceStorage> NonMutableModule for LayerNorm1D<M, E, D> {}

impl<const M: usize, E: Dtype, D: Device<E>> TensorCollection<E, D> for LayerNorm1D<M, E, D> {
    type To<E2: Dtype, D2: Device<E2>> = LayerNorm1D<M, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::tensor(
                    "gamma",
                    |s| &s.gamma,
                    |s| &mut s.gamma,
                    TensorOptions::reset_to_ones(),
                ),
                Self::tensor(
                    "beta",
                    |s| &s.beta,
                    |s| &mut s.beta,
                    TensorOptions::reset_to_zeros(),
                ),
            ),
            |(gamma, beta)| LayerNorm1D {
                gamma,
                beta,
                epsilon: V::E2::from_f32(1e-5).unwrap(),
            },
        )
    }
}

impl<const M: usize, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<Rank1<M>, E, D, T>>
    for LayerNorm1D<M, E, D>
{
    type Output = Tensor<Rank1<M>, E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, x: Tensor<Rank1<M>, E, D, T>) -> Result<Self::Output, D::Err> {
        x.try_normalize(self.epsilon)?
            .try_mul(self.gamma.clone())?
            .try_add(self.beta.clone())
    }
}

impl<B: Dim, const M: usize, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(B, Const<M>), E, D, T>> for LayerNorm1D<M, E, D>
{
    type Output = Tensor<(B, Const<M>), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, x: Tensor<(B, Const<M>), E, D, T>) -> Result<Self::Output, D::Err> {
        let shape = *x.shape();
        x.try_normalize::<Axis<1>>(self.epsilon)?
            .try_mul(self.gamma.retaped::<T>().try_broadcast_like(&shape)?)?
            .try_add(self.beta.retaped::<T>().try_broadcast_like(&shape)?)
    }
}

impl<B: Dim, S: Dim, const M: usize, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(B, S, Const<M>), E, D, T>> for LayerNorm1D<M, E, D>
{
    type Output = Tensor<(B, S, Const<M>), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, x: Tensor<(B, S, Const<M>), E, D, T>) -> Result<Self::Output, D::Err> {
        let shape = *x.shape();
        x.try_normalize::<Axis<2>>(self.epsilon)?
            .try_mul(self.gamma.retaped::<T>().try_broadcast_like(&shape)?)?
            .try_add(self.beta.retaped::<T>().try_broadcast_like(&shape)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_layer_norm_reset() {
        let dev: TestDevice = Default::default();

        let mut m = dev.build_module::<builder::LayerNorm1D<5>, TestDtype>();
        assert_eq!(m.gamma.array(), [1.0; 5]);
        assert_eq!(m.beta.array(), [0.0; 5]);

        m.gamma = dev.sample_normal();
        m.beta = dev.sample_normal();

        assert_ne!(m.gamma.array(), [1.0; 5]);
        assert_ne!(m.beta.array(), [0.0; 5]);

        m.reset_params();

        assert_eq!(m.gamma.array(), [1.0; 5]);
        assert_eq!(m.beta.array(), [0.0; 5]);
    }

    #[test]
    fn test_layer_norm_1d_forward() {
        let dev: TestDevice = Default::default();
        let mut m = dev.build_module::<builder::LayerNorm1D<5>, TestDtype>();
        let x = dev.sample_normal::<Rank1<5>>();
        let r = m.forward_mut(x.leaky_trace());
        assert_close(
            &r.array(),
            &[0.873304, 0.9879816, -1.6083492, 0.44028836, -0.6932247],
        );
        let g = r.mean().backward();
        assert_close(
            &g.get(&m.gamma).array(),
            &[0.1746608, 0.19759633, -0.32166985, 0.088057674, -0.13864495],
        );
        assert_close(&g.get(&m.beta).array(), &[0.2; 5]);
    }

    #[test]
    fn test_layer_norm_2d_forward() {
        let dev: TestDevice = Default::default();
        let m = dev.build_module::<builder::LayerNorm1D<5>, TestDtype>();
        let x = dev.sample_normal::<Rank2<3, 5>>();
        let r = m.forward(x.leaky_trace());
        assert_close(
            &r.array(),
            &[
                [0.873304, 0.9879816, -1.6083492, 0.44028836, -0.6932247],
                [0.663322, -1.8449169, 0.05217871, 0.056903206, 1.0725129],
                [1.0343355, -1.5559655, -0.40086073, 1.1405537, -0.21806297],
            ],
        );
        let g = r.mean().backward();
        assert_close(
            &g.get(&m.gamma).array(),
            &[0.1713974, -0.16086, -0.1304687, 0.109183, 0.0107483],
        );
        assert_close(&g.get(&m.beta).array(), &[0.2; 5]);
    }
}
