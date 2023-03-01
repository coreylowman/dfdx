mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{gradients::Tape, shapes::*, tensor::Tensor};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct HardswishKernelOp;

/// [Hard Swish](https://paperswithcode.com/method/hard-swish). `h-swish(x)=x*(ReLU`
///
///

pub fn hardswish<S: Shape, E: Dtype, D: UnaryKernel<HardswishKernelOp, E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.hardswish()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<HardswishKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn hardswish(self) -> Self {
        self.try_hardswish().unwrap()
    }
    pub fn try_hardswish(self) -> Result<Self, D::Err> {
        try_unary_op(HardswishKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tests::*};

    #[test]
    fn test_hardswish() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([
            -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
        ]);
        let r = x.trace().hardswish();
        assert_close(
            &r.array(),
            &[
                0.0, 0.0, -0.333333, -0.375, -0.333333, -0.208333, 0.0, 0.291667, 0.666666, 1.125,
                1.666667, 3.0, 4.0,
            ],
        );
    }
}
