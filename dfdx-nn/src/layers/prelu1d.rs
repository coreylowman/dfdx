use dfdx::{
    shapes::{Dim, Dtype},
    tensor::{Tape, Tensor},
    tensor_ops::{BroadcastTo, Device, TryMul, TryPReLU},
};

use crate::*;

/// Calls [dfdx::tensor_ops::prelu()] with learnable values along second dimension.
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
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, <D>::Err> {
        let a = device
            .try_ones_like(&(self.c,))?
            .try_mul(E::from_f64(self.a).unwrap())?;
        Ok(PReLU1D { a })
    }
}

/// See [PReLU1DConfig].
#[derive(Clone, Debug, UpdateParams, ZeroGrads, SaveSafeTensors, LoadSafeTensors)]
pub struct PReLU1D<C: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[serialize]
    pub a: Tensor<(C,), Elem, Dev>,
}

impl<C: Dim, E: Dtype, D: Device<E>> ResetParams<E, D> for PReLU1D<C, E, D> {
    /// Does nothing.
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        Ok(())
    }
}

impl<C: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(C,), E, D, T>>
    for PReLU1D<C, E, D>
{
    type Output = Tensor<(C,), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, x: Tensor<(C,), E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_prelu(self.a.clone())
    }
}

impl<B: Dim, C: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(B, C), E, D, T>>
    for PReLU1D<C, E, D>
{
    type Output = Tensor<(B, C), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, x: Tensor<(B, C), E, D, T>) -> Result<Self::Output, Self::Error> {
        let a = self.a.retaped::<T>().broadcast_like(&x);
        x.try_prelu(a)
    }
}

impl<B: Dim, C: Dim, H: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(B, C, H), E, D, T>> for PReLU1D<C, E, D>
{
    type Output = Tensor<(B, C, H), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, x: Tensor<(B, C, H), E, D, T>) -> Result<Self::Output, Self::Error> {
        let a = self.a.retaped::<T>().broadcast_like(&x);
        x.try_prelu(a)
    }
}

impl<B: Dim, C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(B, C, H, W), E, D, T>> for PReLU1D<C, E, D>
{
    type Output = Tensor<(B, C, H, W), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, x: Tensor<(B, C, H, W), E, D, T>) -> Result<Self::Output, Self::Error> {
        let a = self.a.retaped::<T>().broadcast_like(&x);
        x.try_prelu(a)
    }
}