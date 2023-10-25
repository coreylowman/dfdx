use crate::prelude::*;

use std::ops::Mul;

/// **Requires Nightly** Flattens 3d tensors to 1d, and 4d tensors to 2d.
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

    fn try_forward(
        &self,
        input: Tensor<(C, H, W), E, D, T>,
    ) -> Result<Self::Output, crate::tensor::Error> {
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

    fn try_forward(
        &self,
        input: Tensor<(Batch, C, H, W), E, D, T>,
    ) -> Result<Self::Output, crate::tensor::Error> {
        let (batch, c, h, w) = *input.shape();
        let dst = (batch, c * h * w);
        input.try_reshape_like(&dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_flattens() {
        let dev: TestDevice = Default::default();
        let _: Tensor<Rank1<100>, TestDtype, _> =
            Flatten2D.forward_mut(dev.zeros::<Rank3<10, 5, 2>>());
        let _: Tensor<Rank2<5, 24>, TestDtype, _> =
            Flatten2D.forward_mut(dev.zeros::<Rank4<5, 4, 3, 2>>());
        let x: Tensor<(usize, Const<4>, Const<3>, Const<2>), TestDtype, _> =
            dev.zeros_like(&(5, Const::<4>, Const::<3>, Const::<2>));
        let y = Flatten2D.forward_mut(x);
        assert_eq!(y.shape(), &(5, Const::<24>));
    }
}
