mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    gradients::Tape,
    tensor::Tensor,
};

use super::{ops::try_unary_op, Device};

#[derive(Debug, Clone, Copy)]
pub struct PowKernelOp<E>(E);

/// Raises to a float power. `t^i`.
///
/// The derivative is `i * t.powf(i - 1)`.
///
/// **Related functions**: [powi()], [sqrt()]
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.powf(-3.2);
/// ```
pub fn powf<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
    exponent: E,
) -> Tensor<S, E, D, T> {
    t.powf(exponent)
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn powf(self, exponent: E) -> Self {
        self.try_powf(exponent).unwrap()
    }
    pub fn try_powf(self, exponent: E) -> Result<Self, D::Err> {
        try_unary_op(PowKernelOp(exponent), self)
    }
}

/// Raises to an integer power. `t^i`.
///
/// The derivative is `i * t.powi(i - 1)`.
///
/// **Related functions**: [powf()], [square()]
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.powi(3);
/// ```
pub fn powi<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
    exponent: i32,
) -> Tensor<S, E, D, T> {
    t.powi(exponent)
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn powi(self, exponent: i32) -> Self {
        self.try_powi(exponent).unwrap()
    }
    pub fn try_powi(self, exponent: i32) -> Result<Self, D::Err> {
        try_unary_op(PowKernelOp(exponent), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        devices::AsArray, tensor::TensorFromArray, tensor_ops::*, tests::build_test_device,
    };

    #[test]
    fn test_powf_positive() {
        let dev = build_test_device!();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.trace().powf(3.5);
        let r_array = r.as_array();
        assert!(r_array[0].is_nan());
        assert!(r_array[1].is_nan());
        assert_eq!(&r_array[2..], &[0.0, 1.0, 11.313708]);

        let g = r.sum().backward();
        let grad = g.get(&t).as_array();
        assert!(grad[0].is_nan());
        assert!(grad[1].is_nan());
        assert_eq!(&grad[2..], &[0.0, 3.5, 19.79899]);
    }

    #[test]
    fn test_powf_negative() {
        let dev = build_test_device!();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.trace().powf(-1.2);
        let r_array = r.as_array();
        assert!(r_array[0].is_nan());
        assert!(r_array[1].is_nan());
        assert_eq!(&r_array[2..], &[f32::INFINITY, 1.0, 0.43527526]);

        let g = r.sum().backward();
        let grad = g.get(&t).as_array();
        assert!(grad[0].is_nan());
        assert!(grad[1].is_nan());
        assert_eq!(&grad[2..], &[f32::NEG_INFINITY, -1.2, -0.26116517]);
    }

    #[test]
    fn test_powi_positive() {
        let dev = build_test_device!();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.trace().powi(3);
        assert_eq!(r.as_array(), [-8., -1., 0., 1., 8.]);
        let g = r.sum().backward();
        assert_eq!(g.get(&t).as_array(), [12., 3., 0., 3., 12.]);
    }

    #[test]
    fn test_powi_negative() {
        let dev = build_test_device!();
        let t = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.trace().powi(-3);
        assert_eq!(r.as_array(), [-0.125, -1.0, f32::INFINITY, 1.0, 0.125]);
        let g = r.sum().backward();
        assert_eq!(
            g.get(&t).as_array(),
            [-0.1875, -3., f32::NEG_INFINITY, -3., -0.1875]
        );
    }
}
