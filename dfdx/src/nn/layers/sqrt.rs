use crate::prelude::*;

/// Calls [dfdx::tensor_ops::sqrt()].
#[derive(Default, Debug, Clone, Copy, CustomModule)]
pub struct Sqrt;

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for Sqrt {
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_sqrt()
    }
}
