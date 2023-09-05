use crate::{CustomModule, Module};
use dfdx::{
    shapes::{Dtype, Shape},
    tensor::{Tape, Tensor},
    tensor_ops::{Device, ReshapeTo},
};

/// Reshapes input tensors to a configured shape.
///
/// Example usage:
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx_nn::*;
/// # let dev: Cpu = Default::default();
/// let model: Reshape<Rank2<5, 24>> = Default::default();
/// let x: Tensor<Rank4<5, 4, 3, 2>, f32, _> = dev.sample_normal();
/// let _: Tensor<Rank2<5, 24>, f32, _> = model.forward(x);
/// ```
#[derive(Default, Debug, Clone, Copy, CustomModule)]
pub struct Reshape<S: Shape>(pub S);

impl<Src: Shape, Dst: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<Src, E, D, T>>
    for Reshape<Dst>
{
    type Output = Tensor<Dst, E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<Src, E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_reshape_like(&self.0)
    }
}
