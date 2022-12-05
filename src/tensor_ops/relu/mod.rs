mod cpu_kernel;

use super::{ops::try_unary_op, Device};
use crate::{gradients::Tape, shapes::*, tensor::Tensor};

#[derive(Debug, Default, Copy, Clone)]
pub struct ReLUKernelOp;

/// [Rectified Linear Unit (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)). `max(0, t)`
///
/// The derivative is the [Heaviside](https://en.wikipedia.org/wiki/Heaviside_step_function) function.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = relu(t.clone());
///
/// // or the tensor method!
/// let r2 = t.relu();
/// ```
pub fn relu<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.relu()
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn relu(self) -> Self {
        self.try_relu().unwrap()
    }
    pub fn try_relu(self) -> Result<Self, D::Err> {
        try_unary_op(ReLUKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::storage::AsArray, tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_relu() {
        let dev = build_test_device!();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().relu();
        assert_eq!(r.array(), [0.0, 0.0, 0.0, 1.0, 2.0]);
        // NOTE: call .exp() to make sure we cover cases where .relu() uses the result's gradient
        let g = r.exp().mean().backward();
        assert_eq!(g.get(&x).array(), [0.0, 0.0, 0.0, 0.54365635, 1.4778112]);
    }
}
