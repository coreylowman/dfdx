mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{Device, HasErr},
    gradients::Tape,
    tensor::Tensor,
};

use super::utils::{try_unary_op, UnaryKernel};

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
pub trait TrySin: HasErr {
    fn sin(self) -> Self {
        self.try_sin().unwrap()
    }
    fn try_sin(self) -> Result<Self, Self::Err>;
}

#[derive(Debug, Default, Copy, Clone)]
pub(super) struct SinKernelOp;

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> TrySin for Tensor<S, E, D, T>
where
    D: UnaryKernel<SinKernelOp, S, S, E>,
{
    fn try_sin(self) -> Result<Self, Self::Err> {
        try_unary_op(Default::default(), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{assert_close, build_test_device};
    use crate::{devices::AsArray, tensor::*, tensor_ops::*};

    #[test]
    fn test_sin() {
        let dev = build_test_device!();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().sin();
        assert_close(
            &r.as_array(),
            &[-0.9092974, -0.84147096, 0.0, 0.84147096, 0.9092974],
        );
        let g = r.mean().backward();
        assert_close(
            &g.get(&x).as_array(),
            &[-0.08322937, 0.10806046, 0.2, 0.10806046, -0.08322937],
        );
    }
}
