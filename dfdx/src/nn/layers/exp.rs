use crate::prelude::*;

/// Calls [dfdx::tensor_ops::exp()].
#[derive(Default, Debug, Clone, Copy, crate::CustomModule)]
pub struct Exp;
impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for Exp {
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_exp()
    }
}
