#[allow(unused)]
use crate::{gradients::Tape, shapes::*, tensor::Tensor, tensor_ops::*};

use super::traits::*;

/// **Requires Nightly** Flattens 3d tensors to 1d, and 4d tensors to 2d.
#[derive(Default, Clone, Copy)]
pub struct Flatten2D;

impl ZeroSizedModule for Flatten2D {}
impl NonMutableModule for Flatten2D {}

#[cfg(feature = "nightly")]
impl<const C: usize, const H: usize, const W: usize, D: Device<E>, E: Dtype, T: Tape<E, D>>
    Module<Tensor<Rank3<C, H, W>, E, D, T>> for Flatten2D
where
    Rank3<C, H, W>: HasSameNumelAs<Rank1<{ C * H * W }>>,
{
    type Output = Tensor<Rank1<{ C * H * W }>, E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<Rank3<C, H, W>, E, D, T>) -> Result<Self::Output, D::Err> {
        input.try_reshape()
    }
}

#[cfg(feature = "nightly")]
impl<const B: usize, const C: usize, const H: usize, const W: usize, D, E: Dtype, T>
    Module<Tensor<Rank4<B, C, H, W>, E, D, T>> for Flatten2D
where
    D: Device<E>,
    T: Tape<E, D>,
    Rank4<B, C, H, W>: HasSameNumelAs<Rank2<B, { C * H * W }>>,
{
    type Output = Tensor<Rank2<B, { C * H * W }>, E, D, T>;
    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<Rank4<B, C, H, W>, E, D, T>,
    ) -> Result<Self::Output, D::Err> {
        input.try_reshape()
    }
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor::ZerosTensor, tests::*};

    #[test]
    fn test_flattens() {
        let dev: TestDevice = Default::default();
        let _: Tensor<Rank1<100>, TestDtype, _> =
            Flatten2D.forward_mut(dev.zeros::<Rank3<10, 5, 2>>());
        let _: Tensor<Rank2<5, 24>, TestDtype, _> =
            Flatten2D.forward_mut(dev.zeros::<Rank4<5, 4, 3, 2>>());
    }
}
