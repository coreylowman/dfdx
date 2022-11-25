mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{Device, HasErr},
    gradients::Tape,
    tensor::Tensor,
};

use super::utils::{try_unary_op, UnaryKernel};

/// [Hyperbolic Tangent (Tanh)](https://en.wikipedia.org/wiki/Hyperbolic_functions).
///
/// The derivative is `1.0 - square(tanh(t))`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = tanh(t.clone());
///
/// // or the tensor method!
/// let r2 = t.tanh();
/// ```
pub trait TryTanh: HasErr {
    fn tanh(self) -> Self {
        self.try_tanh().unwrap()
    }
    fn try_tanh(self) -> Result<Self, Self::Err>;
}

#[derive(Debug, Default, Copy, Clone)]
pub(super) struct TanhKernelOp;

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> TryTanh for Tensor<S, E, D, T>
where
    D: UnaryKernel<TanhKernelOp, S, S, E>,
{
    fn try_tanh(self) -> Result<Self, Self::Err> {
        try_unary_op(Default::default(), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{devices::AsArray, tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_tanh() {
        let dev = build_test_device!();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().tanh();
        assert_eq!(
            r.as_array(),
            [-0.9640276, -0.7615942, 0., 0.7615942, 0.9640276]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.get(&x).as_array(),
            [0.014130163, 0.083994865, 0.2, 0.083994865, 0.014130163]
        );
    }
}
