mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{Device, HasErr},
    gradients::Tape,
    tensor::Tensor,
};

use super::utils::{try_unary_op, UnaryKernel};

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
pub trait Sigmoid: HasErr {
    fn sigmoid(self) -> Self {
        self.try_sigmoid().unwrap()
    }
    fn try_sigmoid(self) -> Result<Self, Self::Err>;
}

#[derive(Debug, Default, Copy, Clone)]
pub(super) struct SigmoidKernelOp;

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> Sigmoid for Tensor<S, E, D, T>
where
    D: UnaryKernel<SigmoidKernelOp, S, S, E>,
{
    fn try_sigmoid(self) -> Result<Self, Self::Err> {
        try_unary_op(Default::default(), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{devices::AsArray, tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_sigmoid() {
        let dev = build_test_device!();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().sigmoid();
        assert_eq!(
            r.as_array(),
            [0.11920292, 0.26894143, 0.5, 0.7310586, 0.880797]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.get(&x).as_array(),
            [0.020998716, 0.039322387, 0.05, 0.039322387, 0.020998726]
        );
    }
}
