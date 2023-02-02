#[allow(unused)]
use crate::{gradients::Tape, shapes::*, tensor::Tensor, tensor_ops::*};

#[allow(unused)]
use super::{BuildModule, Module, NonMutableModule, ZeroSizedModule};

/// **Requires Nightly** Flattens 3d tensors to 1d, and 4d tensors to 2d.
#[derive(Default, Clone, Copy)]
pub struct Flatten2D;

impl ZeroSizedModule for Flatten2D {}
impl NonMutableModule for Flatten2D {}

impl<D: Device<E>, E: Dtype> BuildModule<D, E> for Flatten2D {
    type Built = Self;
    fn try_build(_: &D) -> Result<Self, <D>::Err> {
        Ok(Default::default())
    }
}

#[cfg(feature = "nightly")]
impl<const C: usize, const H: usize, const W: usize, D: Device<E>, E: Dtype, T: Tape<D>>
    Module<Tensor<Rank3<C, H, W>, E, D, T>> for Flatten2D
where
    Rank3<C, H, W>: HasSameNumelAs<Rank1<{ C * H * W }>>,
{
    type Output = Tensor<Rank1<{ C * H * W }>, E, D, T>;
    fn forward(&self, input: Tensor<Rank3<C, H, W>, E, D, T>) -> Self::Output {
        input.reshape()
    }
}

#[cfg(feature = "nightly")]
impl<const B: usize, const C: usize, const H: usize, const W: usize, D, E: Dtype, T: Tape<D>>
    Module<Tensor<Rank4<B, C, H, W>, E, D, T>> for Flatten2D
where
    D: Device<E>,
    Rank4<B, C, H, W>: HasSameNumelAs<Rank2<B, { C * H * W }>>,
{
    type Output = Tensor<Rank2<B, { C * H * W }>, E, D, T>;
    fn forward(&self, input: Tensor<Rank4<B, C, H, W>, E, D, T>) -> Self::Output {
        input.reshape()
    }
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nn::ModuleMut, tensor::ZerosTensor, tests::TestDevice};

    #[test]
    fn test_flattens() {
        let dev: TestDevice = Default::default();
        let _: Tensor<Rank1<100>, _, _> = Flatten2D.forward_mut(dev.zeros::<Rank3<10, 5, 2>>());
        let _: Tensor<Rank2<5, 24>, _, _> = Flatten2D.forward_mut(dev.zeros::<Rank4<5, 4, 3, 2>>());
    }
}
