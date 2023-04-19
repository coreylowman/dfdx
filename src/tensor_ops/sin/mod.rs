mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct SinKernelOp;

/// [Sine function](https://en.wikipedia.org/wiki/Sine_and_cosine).
///
/// It's derivative is `cos(t)`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.sin();
/// ```
pub fn sin<S: Shape, E: Dtype, D: UnaryKernel<SinKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.sin()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<SinKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [sin]
    pub fn sin(self) -> Self {
        self.try_sin().unwrap()
    }
    /// See [sin]
    pub fn try_sin(self) -> Result<Self, D::Err> {
        try_unary_op(SinKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::*;
    use crate::{tensor::*, tensor_ops::*};

    #[test]
    fn test_sin() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().sin();
        assert_close_to_literal!(r, [-0.9092974, -0.84147096, 0.0, 0.84147096, 0.9092974]);
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [-0.08322937, 0.10806046, 0.2, 0.10806046, -0.08322937]
        );
    }
}
