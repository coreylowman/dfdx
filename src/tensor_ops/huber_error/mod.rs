mod cpu_kernel;

use super::{ops::try_binary_op, Device};
use crate::{gradients::*, shapes::*, tensor::Tensor};

#[derive(Debug, Default, Clone, Copy)]
pub struct HuberErrorKernelOp<E: Dtype> {
    pub delta: E,
}

/// TODO docstring
pub fn huber_error<S: Shape, E: Dtype, D: Device<E>, T: Tape<D> + Merge<R>, R: Tape<D>>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, R>,
    delta: E,
) -> Tensor<S, E, D, T> {
    lhs.huber_error(rhs, delta)
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    /// See [huber_error]
    pub fn huber_error<R: Tape<D>>(self, rhs: Tensor<S, E, D, R>, delta: E) -> Self
    where
        T: Merge<R>,
    {
        self.try_huber_error(rhs, delta).unwrap()
    }

    /// See [huber_error]
    pub fn try_huber_error<R: Tape<D>>(
        self,
        rhs: Tensor<S, E, D, R>,
        delta: E,
    ) -> Result<Self, D::Err>
    where
        T: Merge<R>,
    {
        try_binary_op(HuberErrorKernelOp { delta }, self, rhs)
    }
}
