use crate::prelude::*;

/// Calls [crate::tensor_ops::cos()].
#[derive(Default, Debug, Clone, Copy, CustomModule)]
pub struct Cos;
impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for Cos {
    type Output = Tensor<S, E, D, T>;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        x.try_cos()
    }
}
