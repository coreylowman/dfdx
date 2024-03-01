use crate::prelude::*;

/// Calls [crate::tensor_ops::prelu()] with learnable values along second dimension.
#[derive(Debug, Clone, Copy)]
pub struct PReLU1DConfig<C: Dim> {
    pub a: f64,
    pub c: C,
}

impl<C: Dim + Default> Default for PReLU1DConfig<C> {
    fn default() -> Self {
        Self {
            a: 0.25,
            c: Default::default(),
        }
    }
}

impl<C: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for PReLU1DConfig<C> {
    type Built = PReLU1D<C, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        let a = device.try_ones_like(&(self.c,))?.try_mul(self.a)?;
        Ok(PReLU1D { a })
    }
}

/// See [PReLU1DConfig].
#[derive(Clone, Debug, UpdateParams, ZeroGrads, WithGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct PReLU1D<C: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub a: Tensor<(C,), Elem, Dev>,
}

impl<C: Dim, E: Dtype, D: Device<E>> ResetParams<E, D> for PReLU1D<C, E, D> {
    /// Does nothing.
    fn try_reset_params(&mut self) -> Result<(), crate::tensor::Error> {
        Ok(())
    }
}

impl<C: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(C,), E, D, T>>
    for PReLU1D<C, E, D>
{
    type Output = Tensor<(C,), E, D, T>;
    fn try_forward(&self, x: Tensor<(C,), E, D, T>) -> Result<Self::Output, Error> {
        x.try_prelu(self.a.clone())
    }
}

impl<B: Dim, C: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(B, C), E, D, T>>
    for PReLU1D<C, E, D>
{
    type Output = Tensor<(B, C), E, D, T>;
    fn try_forward(&self, x: Tensor<(B, C), E, D, T>) -> Result<Self::Output, Error> {
        let a = self.a.retaped::<T>().broadcast_like(&x);
        x.try_prelu(a)
    }
}

impl<B: Dim, C: Dim, H: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(B, C, H), E, D, T>> for PReLU1D<C, E, D>
{
    type Output = Tensor<(B, C, H), E, D, T>;
    fn try_forward(&self, x: Tensor<(B, C, H), E, D, T>) -> Result<Self::Output, Error> {
        let a = self.a.retaped::<T>().broadcast_like(&x);
        x.try_prelu(a)
    }
}

impl<B: Dim, C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(B, C, H, W), E, D, T>> for PReLU1D<C, E, D>
{
    type Output = Tensor<(B, C, H, W), E, D, T>;
    fn try_forward(&self, x: Tensor<(B, C, H, W), E, D, T>) -> Result<Self::Output, Error> {
        let a = self.a.retaped::<T>().broadcast_like(&x);
        x.try_prelu(a)
    }
}
