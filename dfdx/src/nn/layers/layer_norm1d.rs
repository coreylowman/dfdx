use crate::prelude::*;

/// Implements layer normalization as described in [Layer Normalization](https://arxiv.org/abs/1607.06450).
///
/// This calls [normalize()] on the last axis of the input to normalize to 0 mean and unit std dev, and then does an element-wise
/// affine transform using learnable parameters.
///
/// Epsilon is passed to [normalize()] and added to the variance to ensure big enough numbers. It defaults to `1e-5`.
///
/// Generics:
/// - `M` The size of the affine transform tensors.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx::*;
/// # let dev: Cpu = Default::default();
/// type Model = LayerNorm1DConstConfig<5>;
/// let model = dev.build_module::<f32>(Model::default());
/// let _: Tensor<Rank1<5>, f32, _> = model.forward(dev.zeros::<Rank1<5>>());
/// ```
#[derive(Default, Clone, Copy, Debug)]
#[repr(transparent)]
pub struct LayerNorm1DConfig<M: Dim>(pub M);

/// Compile time sugar alias around [LayerNorm1DConfig]
pub type LayerNorm1DConstConfig<const M: usize> = LayerNorm1DConfig<Const<M>>;

impl<M: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for LayerNorm1DConfig<M> {
    type Built = LayerNorm1D<M, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(LayerNorm1D {
            gamma: device.try_ones_like(&(self.0,))?,
            beta: device.try_zeros_like(&(self.0,))?,
            epsilon: 1e-5,
        })
    }
}

/// See [LayerNorm1DConfig]
#[derive(Clone, Debug, UpdateParams, ZeroGrads, WithGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct LayerNorm1D<M: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub gamma: Tensor<(M,), Elem, Dev>,
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub beta: Tensor<(M,), Elem, Dev>,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub epsilon: f64,
}

impl<M: Dim, E: Dtype, D: Device<E>> ResetParams<E, D> for LayerNorm1D<M, E, D> {
    fn try_reset_params(&mut self) -> Result<(), crate::tensor::Error> {
        self.gamma.try_fill_with_ones()?;
        self.beta.try_fill_with_zeros()
    }
}

impl<M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(M,), E, D, T>>
    for LayerNorm1D<M, E, D>
{
    type Output = Tensor<(M,), E, D, T>;
    fn try_forward(&self, x: Tensor<(M,), E, D, T>) -> Result<Self::Output, Error> {
        x.try_normalize(self.epsilon)?
            .try_mul(self.gamma.clone())?
            .try_add(self.beta.clone())
    }
}

impl<Batch: Dim, M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(Batch, M), E, D, T>>
    for LayerNorm1D<M, E, D>
{
    type Output = Tensor<(Batch, M), E, D, T>;
    fn try_forward(&self, x: Tensor<(Batch, M), E, D, T>) -> Result<Self::Output, Error> {
        let x = x.try_normalize::<Axis<1>>(self.epsilon)?;
        let x = self.gamma.retaped::<T>().broadcast_like(&x).try_mul(x)?;
        self.beta.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}

impl<Batch: Dim, Seq: Dim, M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(Batch, Seq, M), E, D, T>> for LayerNorm1D<M, E, D>
{
    type Output = Tensor<(Batch, Seq, M), E, D, T>;
    fn try_forward(&self, x: Tensor<(Batch, Seq, M), E, D, T>) -> Result<Self::Output, Error> {
        let x = x.try_normalize::<Axis<2>>(self.epsilon)?;
        let x = self.gamma.retaped::<T>().broadcast_like(&x).try_mul(x)?;
        self.beta.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_layer_norm_reset() {
        let dev: TestDevice = Default::default();

        let mut m = dev.build_module::<TestDtype>(<LayerNorm1DConstConfig<5>>::default());
        assert_close_to_literal!(m.gamma, [1.0; 5]);
        assert_close_to_literal!(m.beta, [0.0; 5]);

        m.gamma = dev.sample_normal();
        m.beta = dev.sample_normal();

        assert_ne!(m.gamma.array(), [TestDtype::ONE; 5]);
        assert_ne!(m.beta.array(), [TestDtype::default(); 5]);

        m.reset_params();

        assert_close_to_literal!(m.gamma, [1.0; 5]);
        assert_close_to_literal!(m.beta, [0.0; 5]);
    }

    #[test]
    fn test_layer_norm_1d_forward() {
        let dev: TestDevice = Default::default();
        let mut m = dev.build_module::<TestDtype>(<LayerNorm1DConstConfig<5>>::default());
        let x = dev.sample_normal::<Rank1<5>>();
        let r = m.forward_mut(x.leaky_trace());
        assert_close_to_literal!(r, [0.873304, 0.9879816, -1.6083492, 0.44028836, -0.6932247]);
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&m.gamma),
            [0.1746608, 0.19759633, -0.32166985, 0.088057674, -0.13864495]
        );
        assert_close_to_literal!(g.get(&m.beta), [0.2; 5]);
    }

    #[test]
    fn test_layer_norm_2d_forward() {
        let dev: TestDevice = Default::default();
        let m = dev.build_module::<TestDtype>(<LayerNorm1DConstConfig<5>>::default());
        let x = dev.sample_normal::<Rank2<3, 5>>();
        let r = m.forward(x.leaky_trace());
        assert_close_to_literal!(
            r,
            [
                [0.873304, 0.9879816, -1.6083492, 0.44028836, -0.6932247],
                [0.663322, -1.8449169, 0.05217871, 0.056903206, 1.0725129],
                [1.0343355, -1.5559655, -0.40086073, 1.1405537, -0.21806297],
            ]
        );
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&m.gamma),
            [0.1713974, -0.16086, -0.1304687, 0.109183, 0.0107483]
        );
        assert_close_to_literal!(g.get(&m.beta), [0.2; 5]);
    }
}
