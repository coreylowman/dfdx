mod cpu_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{gradients::Tape, shapes::*, tensor::Tensor};

#[derive(Debug, Clone, Copy)]
pub struct PowKernelOp<E>(E);

/// Raises to a float power; `t^i`.
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.powf(-3.2);
/// ```
pub fn powf<S: Shape, E: Dtype, D: UnaryKernel<PowKernelOp<E>, E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
    exponent: E,
) -> Tensor<S, E, D, T> {
    t.powf(exponent)
}

impl<S: Shape, E: Dtype, D: UnaryKernel<PowKernelOp<E>, E>, T: Tape<D>> Tensor<S, E, D, T> {
    /// See [powf]
    pub fn powf(self, exponent: E) -> Self {
        self.try_powf(exponent).unwrap()
    }
    /// See [powf]
    pub fn try_powf(self, exponent: E) -> Result<Self, D::Err> {
        try_unary_op(PowKernelOp(exponent), self)
    }
}

/// Raises to an integer power; `t^i`.
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.powi(3);
/// ```
pub fn powi<S: Shape, E: Dtype, D: UnaryKernel<PowKernelOp<i32>, E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
    exponent: i32,
) -> Tensor<S, E, D, T> {
    t.powi(exponent)
}

impl<S: Shape, E: Dtype, D: UnaryKernel<PowKernelOp<i32>, E>, T: Tape<D>> Tensor<S, E, D, T> {
    /// See [powi]
    pub fn powi(self, exponent: i32) -> Self {
        self.try_powi(exponent).unwrap()
    }
    /// See [powi]
    pub fn try_powi(self, exponent: i32) -> Result<Self, D::Err> {
        try_unary_op(PowKernelOp(exponent), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::TestDevice};

    #[test]
    fn test_powf_positive() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.trace().powf(3.5);
        let r_array = r.array();
        assert!(r_array[0].is_nan());
        assert!(r_array[1].is_nan());
        assert_eq!(&r_array[2..], &[0.0, 1.0, 11.313708]);

        let g = r.sum().backward();
        let grad = g.get(&t).array();
        assert!(grad[0].is_nan());
        assert!(grad[1].is_nan());
        assert_eq!(&grad[2..], &[0.0, 3.5, 19.79899]);
    }

    #[test]
    fn test_powf_negative() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.trace().powf(-1.2);
        let r_array = r.array();
        assert!(r_array[0].is_nan());
        assert!(r_array[1].is_nan());
        assert_eq!(&r_array[2..], &[f32::INFINITY, 1.0, 0.43527526]);

        let g = r.sum().backward();
        let grad = g.get(&t).array();
        assert!(grad[0].is_nan());
        assert!(grad[1].is_nan());
        assert_eq!(&grad[2..], &[f32::NEG_INFINITY, -1.2, -0.26116517]);
    }

    #[test]
    fn test_powi_positive() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.trace().powi(3);
        assert_eq!(r.array(), [-8., -1., 0., 1., 8.]);
        let g = r.sum().backward();
        assert_eq!(g.get(&t).array(), [12., 3., 0., 3., 12.]);
    }

    #[test]
    fn test_powi_negative() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.trace().powi(-3);
        assert_eq!(r.array(), [-0.125, -1.0, f32::INFINITY, 1.0, 0.125]);
        let g = r.sum().backward();
        assert_eq!(
            g.get(&t).array(),
            [-0.1875, -3., f32::NEG_INFINITY, -3., -0.1875]
        );
    }
}
