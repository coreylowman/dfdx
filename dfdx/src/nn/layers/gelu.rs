use crate::prelude::*;

/// Calls [crate::tensor_ops::fast_gelu()].
#[derive(Default, Debug, Clone, Copy, CustomModule)]
pub struct FastGeLU;
impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for FastGeLU {
    type Output = Tensor<S, E, D, T>;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        x.try_fast_gelu()
    }
}

/// Calls [crate::tensor_ops::accurate_gelu()].
#[derive(Default, Debug, Clone, Copy, CustomModule)]
pub struct AccurateGeLU;
impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for AccurateGeLU {
    type Output = Tensor<S, E, D, T>;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        x.try_accurate_gelu()
    }
}
