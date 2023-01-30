mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::{ops::try_binary_op, Device};
use crate::{gradients::*, shapes::*, tensor::Tensor};

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

#[cfg(test)]
mod tests {
    use crate::{
        tensor::*,
        tests::*,
    };

    #[test]
    fn test_huber_error() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([
            [-0.84240317, 0.63094819, 1.04164326],
            [1.32522500, 0.58402753, 1.91676331],
        ]);
        let b: Tensor<_, TestDtype, _> = dev.tensor([
            [0.52022195, 0.57880402, 0.17535722],
            [0.75429636, 0.66566986, 0.61827511],
        ]);
        let r1 = a.trace().huber_error(b.trace(), 1.0);
        let r2 = a.trace().huber_error(b.trace(), 100.0);
        assert_close(
            &r1.array(),
            &[
                [0.8626251, 0.0013595072, 0.37522575],
                [0.16297975, 0.003332735, 0.79848814],
            ],
        );
        assert_close(
            &r2.array(),
            &((a.clone() - b.clone()).square() / 2.0).array(),
        );
    }
}
