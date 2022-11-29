mod cpu_kernel;

use super::{ops::try_unary_op, Device};
use crate::{arrays::*, gradients::Tape, tensor::Tensor};

#[derive(Debug, Default, Copy, Clone)]
pub struct LnKernelOp;

/// [Natural Logarithm (ln)](https://en.wikipedia.org/wiki/Natural_logarithm). `log_e(t)`.
///
/// It's derivative is `1 / t`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = ln(t.clone());
///
/// // or the tensor method!
/// let r2 = t.ln();
/// ```
pub fn ln<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.ln()
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn ln(self) -> Self {
        self.try_ln().unwrap()
    }
    pub fn try_ln(self) -> Result<Self, D::Err> {
        try_unary_op(LnKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_ln() {
        let dev = build_test_device!();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().ln();
        let r_array = r.as_array();
        assert!(r_array[0].is_nan());
        assert!(r_array[1].is_nan());
        assert!(r_array[2..] == [f32::NEG_INFINITY, 0.0, std::f32::consts::LN_2]);
        let g = r.mean().backward();
        assert_eq!(g.get(&x).as_array(), [-0.1, -0.2, f32::INFINITY, 0.2, 0.1]);
    }
}
