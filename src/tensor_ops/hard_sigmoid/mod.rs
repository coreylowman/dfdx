mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct HardSigmoidKernelOp;

/// [Hard Sigmoid](https://arxiv.org/abs/1905.02244). `relu6(x + 3) / 6`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-4.0, -1.0, 0.0, 1.0, 2.0, 4.0]);
/// let r = t.hard_sigmoid();
/// ```
pub fn hard_sigmoid<S: Shape, E: Dtype, D: UnaryKernel<HardSigmoidKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.hard_sigmoid()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<HardSigmoidKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [hard_sigmoid]
    pub fn hard_sigmoid(self) -> Self {
        self.try_hard_sigmoid().unwrap()
    }
    /// See [hard_sigmoid]
    pub fn try_hard_sigmoid(self) -> Result<Self, D::Err> {
        try_unary_op(HardSigmoidKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_hard_sigmoid() {
        let dev: TestDevice = Default::default();
        let x = dev
            .tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
            .to_dtype::<TestDtype>();
        let r = x.leaky_trace().hard_sigmoid();
        assert_close_to_literal!(r, [0.0, 0.3333333, 0.5, 0.6666666, 1.0]);
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&x), [0.0, 0.033333335, 0.033333335, 0.033333335, 0.0]);
    }
}
