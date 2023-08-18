use dfdx::{
    prelude::{Device, Dim, Dtype, Tape, Tensor},
    shapes::Const,
    tensor_ops::{BroadcastTo, TryAdd},
};

use crate::*;

#[derive(Default, Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bias2DConfig<I: Dim>(pub I);

pub type Bias2DConstConfig<const I: usize> = Bias2DConfig<Const<I>>;

impl<I: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for Bias2DConfig<I> {
    type Built = Bias2D<I, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(Bias2D {
            bias: device.try_zeros_like(&(self.0,))?,
        })
    }
}

#[derive(Clone, Debug, UpdateParams, ZeroGrads, SaveSafeTensors, LoadSafeTensors)]
pub struct Bias2D<I: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[serialize]
    pub bias: Tensor<(I,), Elem, Dev>,
}

impl<I: Dim, E: Dtype, D: Device<E>> ResetParams<E, D> for Bias2D<I, E, D> {
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        self.bias.try_fill_with_zeros()
    }
}

impl<C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(C, H, W), E, D, T>> for Bias2D<W, E, D>
{
    type Output = Tensor<(C, H, W), E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<(C, H, W), E, D, T>) -> Result<Self::Output, Self::Error> {
        self.bias.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}

impl<B: Dim, C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(B, C, H, W), E, D, T>> for Bias2D<W, E, D>
{
    type Output = Tensor<(B, C, H, W), E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<(B, C, H, W), E, D, T>) -> Result<Self::Output, Self::Error> {
        self.bias.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}
