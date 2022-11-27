mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{DeviceStorage, HasErr},
    gradients::{Merge, Tape},
    tensor::Tensor,
};

use super::ops::{try_binary_op, try_unary_op, BinaryKernel, UnaryKernel};

/// Element wise and scalar division.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = tensor([[1.0, 0.5, 1.0], [0.5, 1.0, 3.0]]);
/// let r = div(a, b); // or `a / b`
/// assert_eq!(r.data(), &[[1.0, 4.0, 3.0], [-2.0, -2.0, -1.0]]);
/// ```
///
/// Scalar example:
/// ```rust
/// todo!()
/// ```
pub trait TryDiv<Rhs = Self>: HasErr {
    fn try_div(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct BinaryDivKernelOp;

impl<S: Shape, E: Dtype, D: DeviceStorage, LhsTape: Tape<D>, RhsTape: Tape<D>>
    TryDiv<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<BinaryDivKernelOp, S, S, S, E>,
    LhsTape: Merge<RhsTape>,
{
    fn try_div(self, rhs: Tensor<S, E, D, RhsTape>) -> Result<Self, Self::Err> {
        try_binary_op(Default::default(), self, rhs)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ScalarDivKernelOp<E>(pub(crate) E);

impl<S: Shape, E: Dtype, D: DeviceStorage, T: Tape<D>> TryDiv<E> for Tensor<S, E, D, T>
where
    D: UnaryKernel<ScalarDivKernelOp<E>, S, S, E>,
{
    fn try_div(self, s: E) -> Result<Self, Self::Err> {
        try_unary_op(ScalarDivKernelOp(s), self)
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage, LhsTape: Tape<D>, Rhs> std::ops::Div<Rhs>
    for Tensor<S, E, D, LhsTape>
where
    Self: TryDiv<Rhs>,
{
    type Output = Self;
    fn div(self, rhs: Rhs) -> Self::Output {
        self.try_div(rhs).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::devices::AsArray;
    use crate::tensor::TensorFromArray;
    use crate::tensor_ops::*;
    use crate::tests::build_test_device;

    #[test]
    fn test_div_0d() {
        let dev = build_test_device!();

        let a = dev.tensor(2.0);
        let b = dev.tensor(4.0);

        let r = b.trace() / a.clone();
        assert_eq!(r.as_array(), 2.0);
        let g = r.backward();
        assert_eq!(g.get(&a).as_array(), -1.0);
        assert_eq!(g.get(&b).as_array(), 0.5);
    }

    #[test]
    fn test_div_1d() {
        let dev = build_test_device!();
        let a = dev.tensor([1.0, 2.0, 3.0]);
        let b = dev.tensor([1.0, -1.0, 0.0]);

        let r = b.trace() / a.clone();
        assert_eq!(r.as_array(), [1.0, -0.5, 0.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&a).as_array(), [-1.0 / 3.0, 1.0 / 12.0, 0.0]);
        assert_eq!(g.get(&b).as_array(), [1.0 / 3.0, 1.0 / 6.0, 0.11111112]);
    }

    #[test]
    fn test_div_2d() {
        let dev = build_test_device!();
        let a = dev.tensor([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b = dev.tensor([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let r = b.trace() / a.clone();
        assert_eq!(
            r.as_array(),
            [
                [0.79132426, 2.2505856, 2.5059998],
                [1.4597031, 0.52524966, 0.046511628]
            ]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
            [
                [-0.20074181, -2.1961217, -2.7844446],
                [-0.42998204, -0.12488105, -0.009292662]
            ]
        );
        assert_eq!(
            g.get(&b).as_array(),
            [
                [0.25367835, 0.97580016, 1.1111112],
                [0.29456818, 0.2377556, 0.1997922]
            ]
        );
    }

    #[test]
    fn test_scalar_div_0d() {
        let dev = build_test_device!();
        let x = dev.tensor(1.0);
        let r = x.trace() / 2.0;
        assert_eq!(r.as_array(), 0.5);
        let g = r.exp().backward();
        assert_eq!(g.get(&x).as_array(), 0.8243606);
    }

    #[test]
    fn test_scalar_div_1d() {
        let dev = build_test_device!();
        let x = dev.tensor([0.0, 1.0, 2.0]);
        let r = x.trace() / 2.0;
        assert_eq!(r.as_array(), [0.0, 0.5, 1.0]);
        let g = r.exp().sum().backward();
        assert_eq!(g.get(&x).as_array(), [0.5, 0.8243606, 1.3591409]);
    }

    #[test]
    fn test_scalar_div_2d() {
        let dev = build_test_device!();
        let x = dev.tensor([[1.0; 2]; 3]);
        let r = x.trace() / 2.0;
        assert_eq!(r.as_array(), [[0.5; 2]; 3]);
        let g = r.exp().sum().backward();
        assert_eq!(g.get(&x).as_array(), [[0.8243606; 2]; 3]);
    }
}
