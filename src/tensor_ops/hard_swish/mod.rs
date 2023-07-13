mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct HardSwishKernelOp;

/// [Hard Swish](https://paperswithcode.com/method/hard-swish). `x * (relu6(x + 3) / 6)`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-4.0, -1.0, 0.0, 1.0, 4.0]);
/// let r = t.hard_swish();
/// ```
pub fn hard_swish<S: Shape, E: Dtype, D: UnaryKernel<HardSwishKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.hard_swish()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<HardSwishKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [hard_swish]
    pub fn hard_swish(self) -> Self {
        self.try_hard_swish().unwrap()
    }
    /// See [hard_swish]
    pub fn try_hard_swish(self) -> Result<Self, D::Err> {
        try_unary_op(HardSwishKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_hard_swish() {
        let dev: TestDevice = Default::default();
        let x = dev
            .tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
            .to_dtype::<TestDtype>();
        let r = x.leaky_trace().hard_swish();
        assert_close_to_literal!(r, [0.0, -0.33333334, 0.0, 0.6666667, 4.0]);
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&x), [0.0, 0.033333335, 0.1, 0.16666667, 0.36666667]);
    }
}
