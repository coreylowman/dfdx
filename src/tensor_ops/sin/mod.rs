mod cpu_kernel;

use super::{ops::try_unary_op, Device};
use crate::{gradients::Tape, shapes::*, tensor::Tensor};

#[derive(Debug, Default, Copy, Clone)]
pub struct SinKernelOp;

/// [Sine function](https://en.wikipedia.org/wiki/Sine_and_cosine).
///
/// It's derivative is `cos(t)`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = sin(t.clone());
///
/// // or the tensor method!
/// let r2 = t.sin();
/// ```
pub fn sin<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.sin()
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn sin(self) -> Self {
        self.try_sin().unwrap()
    }
    pub fn try_sin(self) -> Result<Self, D::Err> {
        try_unary_op(SinKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{assert_close, build_test_device};
    use crate::{tensor::storage::AsArray, tensor::*, tensor_ops::*};

    #[test]
    fn test_sin() {
        let dev = build_test_device!();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().sin();
        assert_close(
            &r.array(),
            &[-0.9092974, -0.84147096, 0.0, 0.84147096, 0.9092974],
        );
        let g = r.mean().backward();
        assert_close(
            &g.get(&x).array(),
            &[-0.08322937, 0.10806046, 0.2, 0.10806046, -0.08322937],
        );
    }
}
