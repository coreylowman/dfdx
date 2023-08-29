use dfdx::prelude::{Device, Dtype, Shape, Tape, Tensor};

/// Calls [dfdx::tensor_ops::fast_gelu()].
#[derive(Default, Debug, Clone, Copy, crate::CustomModule)]
pub struct FastGeLU;
impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> crate::Module<Tensor<S, E, D, T>>
    for FastGeLU
{
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_fast_gelu()
    }
}

/// Calls [dfdx::tensor_ops::accurate_gelu()].
#[derive(Default, Debug, Clone, Copy, crate::CustomModule)]
pub struct AccurateGeLU;
impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> crate::Module<Tensor<S, E, D, T>>
    for AccurateGeLU
{
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_accurate_gelu()
    }
}
