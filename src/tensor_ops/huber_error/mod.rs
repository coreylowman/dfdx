mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::{ops::try_binary_op, Device};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct HuberErrorKernelOp<E> {
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
pub fn huber_error<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D> + Merge<R>, R: Tape<E, D>>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, R>,
    delta: impl Into<E>,
) -> Tensor<S, E, D, T> {
    lhs.huber_error(rhs, delta)
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [huber_error]
    pub fn huber_error<R: Tape<E, D>>(self, rhs: Tensor<S, E, D, R>, delta: impl Into<E>) -> Self
    where
        T: Merge<R>,
    {
        self.try_huber_error(rhs, delta).unwrap()
    }

    /// See [huber_error]
    pub fn try_huber_error<R: Tape<E, D>>(
        self,
        rhs: Tensor<S, E, D, R>,
        delta: impl Into<E>,
    ) -> Result<Self, D::Err>
    where
        T: Merge<R>,
    {
        let delta = delta.into();
        try_binary_op(HuberErrorKernelOp { delta }, self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tests::*};

    #[test]
    fn test_huber_error() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([
            [-0.8424031, 0.6309481, 1.0416432],
            [1.325225, 0.5840275, 1.9167633],
        ]);
        let b: Tensor<_, TestDtype, _> = dev.tensor([
            [0.52022195, 0.578804, 0.17535722],
            [0.75429636, 0.66566986, 0.6182751],
        ]);
        let r1 = a.leaky_trace().huber_error(b.leaky_trace(), 1.0);
        let r2 = a.leaky_trace().huber_error(b.leaky_trace(), 100.0);
        assert_close_to_literal!(
            r1,
            [
                [0.8626251, 0.0013595072, 0.37522575],
                [0.16297975, 0.003332735, 0.79848814],
            ]
        );
        assert_close_to_tensor!(r2, (a - b).square() / 2.0);
    }
}
