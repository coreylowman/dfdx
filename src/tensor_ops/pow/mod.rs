mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PowiKernelOp(i32);

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PowfKernelOp<E>(E);

/// Raises to a float power; `t^i`.
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.powf(-3.2);
/// ```
pub fn powf<S: Shape, E: Dtype, D: UnaryKernel<PowfKernelOp<E>, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
    exponent: impl Into<E>,
) -> Tensor<S, E, D, T> {
    t.powf(exponent)
}

impl<S: Shape, E: Dtype, D: UnaryKernel<PowfKernelOp<E>, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [powf]
    pub fn powf(self, exponent: impl Into<E>) -> Self {
        self.try_powf(exponent).unwrap()
    }
    /// See [powf]
    pub fn try_powf(self, exponent: impl Into<E>) -> Result<Self, D::Err> {
        let exponent = exponent.into();
        try_unary_op(PowfKernelOp(exponent), self)
    }
}

/// Raises to an integer power; `t^i`.
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.powi(3);
/// ```
pub fn powi<S: Shape, E: Dtype, D: UnaryKernel<PowiKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
    exponent: i32,
) -> Tensor<S, E, D, T> {
    t.powi(exponent)
}

impl<S: Shape, E: Dtype, D: UnaryKernel<PowiKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [powi]
    pub fn powi(self, exponent: i32) -> Self {
        self.try_powi(exponent).unwrap()
    }
    /// See [powi]
    pub fn try_powi(self, exponent: i32) -> Result<Self, D::Err> {
        try_unary_op(PowiKernelOp(exponent), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_powf_positive() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.leaky_trace().powf(3.5);
        let r_array = r.array();
        assert!(r_array[0].is_nan());
        assert!(r_array[1].is_nan());
        assert_close!(r_array[2], 0.0);
        assert_close!(r_array[3], 1.0);
        assert_close!(r_array[4], 11.313708);

        let g = r.sum().backward();
        let grad = g.get(&t).array();
        assert!(grad[0].is_nan());
        assert!(grad[1].is_nan());
        assert_close!(grad[2], 0.0);
        assert_close!(grad[3], 3.5);
        assert_close!(grad[4], 19.79899);
    }

    #[test]
    fn test_powf_negative() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.leaky_trace().powf(-1.2);
        let r_array = r.array();
        assert!(r_array[0].is_nan());
        assert!(r_array[1].is_nan());
        assert_close!(r_array[2], TestDtype::INFINITY);
        assert_close!(r_array[3], 1.0);
        assert_close!(r_array[4], 0.43527526);

        let g = r.sum().backward();
        let grad = g.get(&t).array();
        assert!(grad[0].is_nan());
        assert!(grad[1].is_nan());
        assert_close!(grad[2], TestDtype::NEG_INFINITY);
        assert_close!(grad[3], -1.2);
        assert_close!(grad[4], -0.26116517);
    }

    #[test]
    fn test_powi_positive() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.leaky_trace().powi(3);
        assert_close_to_literal!(r, [-8., -1., 0., 1., 8.]);
        let g = r.sum().backward();
        assert_close_to_literal!(g.get(&t), [12., 3., 0., 3., 12.]);
    }

    #[test]
    fn test_powi_negative() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.leaky_trace().powi(-3);
        assert_close_to_literal!(r, [-0.125, -1.0, f64::INFINITY, 1.0, 0.125]);
        let g = r.sum().backward();
        assert_close_to_literal!(g.get(&t), [-0.1875, -3., f64::NEG_INFINITY, -3., -0.1875]);
    }
}
