mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{Device, HasErr},
    gradients::Tape,
    tensor::Tensor,
};

use super::ops::{try_unary_op, UnaryKernel};

/// [Cosine function](https://en.wikipedia.org/wiki/Sine_and_cosine).
///
/// It's derivative is `-sin(t)`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = cos(t.clone());
///
/// // or the tensor method!
/// let r2 = t.cos();
/// ```
pub trait TryCos: HasErr {
    fn cos(self) -> Self {
        self.try_cos().unwrap()
    }
    fn try_cos(self) -> Result<Self, Self::Err>;
}

#[derive(Debug, Default, Copy, Clone)]
pub(super) struct CosKernelOp;

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> TryCos for Tensor<S, E, D, T>
where
    D: UnaryKernel<CosKernelOp, S, S, E>,
{
    fn try_cos(self) -> Result<Self, Self::Err> {
        try_unary_op(Default::default(), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{assert_close, build_test_device};
    use crate::{devices::AsArray, tensor::*, tensor_ops::*};

    #[test]
    fn test_cos() {
        let dev = build_test_device!();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().cos();
        assert_close(
            &r.as_array(),
            &[-0.41614684, 0.5403023, 1.0, 0.5403023, -0.41614684],
        );
        let g = r.mean().backward();
        assert_close(
            &g.get(&x).as_array(),
            &[0.18185948, 0.16829419, -0.0, -0.16829419, -0.18185948],
        );
    }
}
