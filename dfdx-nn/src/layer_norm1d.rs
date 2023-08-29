use crate::*;
use dfdx::prelude::*;

/// Implements layer normalization as described in [Layer Normalization](https://arxiv.org/abs/1607.06450).
///
/// This calls [normalize()] on the last axis of the input to normalize to 0 mean and unit std dev, and then does an element-wise
/// affine transform using learnable parameters [Self::gamma] and [Self::beta].
///
/// [Self::epsilon] is passed to [normalize()] and added to the variance to ensure big enough numbers. It defaults to `1e-5`.
///
/// Generics:
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
#[derive(Default, Clone, Copy, Debug)]
#[repr(transparent)]
pub struct LayerNorm1DConfig<M: Dim>(pub M);

/// Compile time sugar alias around [LayerNorm1DConfig]
pub type LayerNorm1DConstConfig<const M: usize> = LayerNorm1DConfig<Const<M>>;

impl<M: Dim, E: Dtype, D: Device<E>> crate::BuildOnDevice<E, D> for LayerNorm1DConfig<M> {
    type Built = LayerNorm1D<M, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(LayerNorm1D {
            gamma: device.try_ones_like(&(self.0,))?,
            beta: device.try_zeros_like(&(self.0,))?,
            epsilon: 1e-5,
        })
    }
}

/// See [LayerNorm1DConfig]
#[derive(Clone, Debug, UpdateParams, ZeroGrads, SaveSafeTensors, LoadSafeTensors)]
pub struct LayerNorm1D<M: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[serialize]
    pub gamma: Tensor<(M,), Elem, Dev>,
    #[param]
    #[serialize]
    pub beta: Tensor<(M,), Elem, Dev>,
    #[serialize]
    pub epsilon: f64,
}

impl<M: Dim, E: Dtype, D: Device<E>> crate::ResetParams<E, D> for LayerNorm1D<M, E, D> {
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        self.gamma.try_fill_with_ones()?;
        self.beta.try_fill_with_zeros()
    }
}

impl<M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> crate::Module<Tensor<(M,), E, D, T>>
    for LayerNorm1D<M, E, D>
{
    type Output = Tensor<(M,), E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<(M,), E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_normalize(self.epsilon)?
            .try_mul(self.gamma.clone())?
            .try_add(self.beta.clone())
    }
}

impl<Batch: Dim, M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::Module<Tensor<(Batch, M), E, D, T>> for LayerNorm1D<M, E, D>
{
    type Output = Tensor<(Batch, M), E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<(Batch, M), E, D, T>) -> Result<Self::Output, Self::Error> {
        let x = x.try_normalize::<Axis<1>>(self.epsilon)?;
        let x = self.gamma.retaped::<T>().broadcast_like(&x).try_mul(x)?;
        self.beta.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}

impl<Batch: Dim, Seq: Dim, M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::Module<Tensor<(Batch, Seq, M), E, D, T>> for LayerNorm1D<M, E, D>
{
    type Output = Tensor<(Batch, Seq, M), E, D, T>;
    type Error = D::Err;
    fn try_forward(
        &self,
        x: Tensor<(Batch, Seq, M), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let x = x.try_normalize::<Axis<2>>(self.epsilon)?;
        let x = self.gamma.retaped::<T>().broadcast_like(&x).try_mul(x)?;
        self.beta.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}
