#[allow(unused)]
use crate::{
    shapes::*,
    tensor::{Tape, Tensor},
    tensor_ops::*,
};

use super::*;

/// Reshapes input tensors to a shape known *at compile time*.
///
/// Example usage:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let model: Reshape<Rank2<5, 24>> = Default::default();
/// let x: Tensor<Rank4<5, 4, 3, 2>, f32, _> = dev.sample_normal();
/// let _: Tensor<Rank2<5, 24>, f32, _> = model.forward(x);
/// ```
#[derive(Default, Clone, Copy)]
pub struct Reshape<S: ConstShape>(S);

impl<S: ConstShape> ZeroSizedModule for Reshape<S> {}
impl<S: ConstShape> NonMutableModule for Reshape<S> {}

impl<Src: Shape, Dst: ConstShape, D: Device<E>, E: Dtype, T: Tape<E, D>>
    Module<Tensor<Src, E, D, T>> for Reshape<Dst>
{
    type Output = Tensor<Dst, E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<Src, E, D, T>) -> Result<Self::Output, D::Err> {
        input.try_reshape_like(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor::ZerosTensor, tests::*};

    #[test]
    fn test_flattens() {
        let dev: TestDevice = Default::default();
        let _: Tensor<Rank1<100>, TestDtype, _> =
            Reshape::<Rank1<100>>::default().forward_mut(dev.zeros::<Rank3<10, 5, 2>>());
        let _: Tensor<Rank2<5, 24>, TestDtype, _> =
            Reshape::<Rank2<5, 24>>::default().forward_mut(dev.zeros::<Rank4<5, 4, 3, 2>>());
        let _: Tensor<Rank3<10, 5, 2>, TestDtype, _> =
            Reshape::<Rank3<10, 5, 2>>::default().forward_mut(dev.zeros::<Rank1<100>>());
        let _: Tensor<Rank4<5, 4, 3, 2>, TestDtype, _> =
            Reshape::<Rank4<5, 4, 3, 2>>::default().forward_mut(dev.zeros::<Rank2<5, 24>>());
    }
}
