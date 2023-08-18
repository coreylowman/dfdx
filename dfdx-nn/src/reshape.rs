use crate::{CustomModule, Module};
use dfdx::{
    shapes::{Dtype, Shape},
    tensor::{Tape, Tensor},
    tensor_ops::{Device, ReshapeTo},
};

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
