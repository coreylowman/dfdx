use super::{Module, ModuleMut, ZeroSizedModule};

use crate::{
    gradients::Tape,
    shapes::*,
    tensor::Tensor,
    tensor_ops::{Device, ReshapeTo},
};

/// **Requires Nightly** Flattens 3d tensors to 1d, and 4d tensors to 2d.
///
/// Specifically:
/// ```ignore
/// # use dfdx::prelude::*;
/// let _: Tensor1D<{3 * 5 * 7}> = Flatten2D.forward(Tensor3D::<3, 5, 7>::zeros());
/// let _: Tensor2D<8, {3 * 5 * 7}> = Flatten2D.forward(Tensor4D::<8, 3, 5, 7>::zeros());
/// ```
#[derive(Default, Clone, Copy)]
pub struct Flatten2D;

impl ZeroSizedModule for Flatten2D {}

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

impl<
        const B: usize,
        const C: usize,
        const H: usize,
        const W: usize,
        D: Device<E>,
        E: Dtype,
        T: Tape<D>,
    > Module<Tensor<Rank4<B, C, H, W>, E, D, T>> for Flatten2D
where
    Rank4<B, C, H, W>: HasSameNumelAs<Rank2<B, { C * H * W }>>,
{
    type Output = Tensor<Rank2<B, { C * H * W }>, E, D, T>;
    fn forward(&self, input: Tensor<Rank4<B, C, H, W>, E, D, T>) -> Self::Output {
        input.reshape()
    }
}

impl<T> ModuleMut<T> for Flatten2D
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;
    fn forward_mut(&mut self, input: T) -> Self::Output {
        self.forward(input)
    }
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor::ZerosTensor, tests::build_test_device};

    #[test]
    fn test_flattens() {
        let dev = build_test_device!();
        let _: Tensor<Rank1<100>, _, _> = Flatten2D.forward_mut(dev.zeros::<Rank3<10, 5, 2>>());
        let _: Tensor<Rank2<5, 24>, _, _> = Flatten2D.forward_mut(dev.zeros::<Rank4<5, 4, 3, 2>>());
    }
}
