mod cpu_kernel;

use super::{ops::try_unary_op, Device};
use crate::{gradients::Tape, shapes::*, tensor::Tensor};

#[derive(Debug, Default, Copy, Clone)]
pub struct SigmoidKernelOp;

/// [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function).
///
/// Equivalent to `1 / (1 + exp(-t))`.
///
/// The derivative is `sigmoid(t) * (1.0 - sigmoid(t))`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = sigmoid(t.clone());
///
/// // or the tensor method!
/// let r2 = t.sigmoid();
/// ```
pub fn sigmoid<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.sigmoid()
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn sigmoid(self) -> Self {
        self.try_sigmoid().unwrap()
    }
    pub fn try_sigmoid(self) -> Result<Self, D::Err> {
        try_unary_op(SigmoidKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::storage::AsArray, tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_sigmoid() {
        let dev = build_test_device!();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().sigmoid();
        assert_eq!(
            r.array(),
            [0.11920292, 0.26894143, 0.5, 0.7310586, 0.880797]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.get(&x).array(),
            [0.020998716, 0.039322387, 0.05, 0.039322387, 0.020998726]
        );
    }
}
