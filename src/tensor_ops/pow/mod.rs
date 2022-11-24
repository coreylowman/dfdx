mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{Device, HasErr},
    gradients::Tape,
    tensor::Tensor,
};

use super::utils::{try_unary_op, UnaryKernel};

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
pub trait Powf<E: Dtype>: HasErr {
    fn powf(self, i: E) -> Self {
        self.try_powf(i).unwrap()
    }
    fn try_powf(self, i: E) -> Result<Self, Self::Err>;
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PowKernelOp<E>(E);

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> Powf<E> for Tensor<S, E, D, T>
where
    D: UnaryKernel<PowKernelOp<E>, S, S, E>,
{
    fn try_powf(self, i: E) -> Result<Self, Self::Err> {
        try_unary_op(PowKernelOp(i), self)
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
pub trait Powi: HasErr {
    fn powi(self, i: i32) -> Self {
        self.try_powi(i).unwrap()
    }
    fn try_powi(self, i: i32) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> Powi for Tensor<S, E, D, T>
where
    D: UnaryKernel<PowKernelOp<i32>, S, S, E>,
{
    fn try_powi(self, i: i32) -> Result<Self, Self::Err> {
        try_unary_op(PowKernelOp(i), self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
