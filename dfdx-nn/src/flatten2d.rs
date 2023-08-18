use std::ops::Mul;

use crate::{CustomModule, Module};

use dfdx::{
    shapes::{Dim, Dtype, HasShape},
    tensor::{Tape, Tensor},
    tensor_ops::{Device, ReshapeTo},
};

#[derive(Debug, Default, Clone, Copy, CustomModule)]
pub struct Flatten2D;

impl<C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(C, H, W), E, D, T>> for Flatten2D
where
    C: Mul<H>,
    <C as Mul<H>>::Output: Mul<W>,
    <<C as Mul<H>>::Output as Mul<W>>::Output: Dim,
{
    type Output = Tensor<(<<C as Mul<H>>::Output as Mul<W>>::Output,), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(C, H, W), E, D, T>) -> Result<Self::Output, D::Err> {
        let (c, h, w) = *input.shape();
        let dst = (c * h * w,);
        input.try_reshape_like(&dst)
    }
}

impl<Batch: Dim, C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(Batch, C, H, W), E, D, T>> for Flatten2D
where
    C: Mul<H>,
    <C as Mul<H>>::Output: Mul<W>,
    <<C as Mul<H>>::Output as Mul<W>>::Output: Dim,
{
    type Output = Tensor<(Batch, <<C as Mul<H>>::Output as Mul<W>>::Output), E, D, T>;
    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(Batch, C, H, W), E, D, T>,
    ) -> Result<Self::Output, D::Err> {
        let (batch, c, h, w) = *input.shape();
        let dst = (batch, c * h * w);
        input.try_reshape_like(&dst)
    }
}
