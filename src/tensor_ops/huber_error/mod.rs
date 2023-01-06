mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::{ops::try_binary_op, Device};
use crate::{gradients::*, shapes::*, tensor::Tensor};

#[derive(Debug, Default, Clone, Copy)]
pub struct HuberErrorKernelOp<E: Dtype> {
    pub delta: E,
}

/// [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss)
/// uses absolute error when the error is higher than `beta`, and squared error when the
/// error is lower than `beta`.
///
/// It computes:
/// 1. if `|x - y| < delta`: `0.5 * (x - y)^2`
/// 2. otherwise: `delta * (|x - y| - 0.5 * delta)`
///
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([1.0, 1.0, 1.0]);
/// let b = dev.tensor([1.5, 1.75, 2.5]);
/// let r = a.huber_error(b, 1.0);
/// assert_eq!(r.array(), [0.125, 0.28125, 1.0]);
/// ```
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
