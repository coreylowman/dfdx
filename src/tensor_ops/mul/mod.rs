mod cpu_kernel;

use super::{ops::*, Device};
use crate::{gradients::*, shapes::*, tensor::*};

#[derive(Debug, Default, Clone, Copy)]
pub struct BinaryMulKernelOp;

#[derive(Debug, Clone, Copy)]
pub struct ScalarMulKernelOp<E>(E);

/// Element wise and scalar multiplication.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = Tensor2D::ones();
/// let r = mul(a, b); // or `a * b`
/// assert_eq!(r.data(), &[[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// ```
///
/// Scalar example:
/// ```rust
/// todo!();
/// ```
pub fn mul<S: Shape, E: Dtype, D: Device<E>, T: Tape<D> + Merge<RhsTape>, RhsTape: Tape<D>>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, RhsTape>,
) -> Tensor<S, E, D, T> {
    lhs * rhs
}
pub trait TryMul<Rhs = Self>: HasErr {
    fn try_mul(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: Device<E>, LhsTape: Tape<D>, RhsTape: Tape<D>>
    TryMul<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    LhsTape: Merge<RhsTape>,
{
    fn try_mul(self, rhs: Tensor<S, E, D, RhsTape>) -> Result<Self, Self::Err> {
        try_binary_op(BinaryMulKernelOp, self, rhs)
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> TryMul<E> for Tensor<S, E, D, T> {
    fn try_mul(self, rhs: E) -> Result<Self, Self::Err> {
        try_unary_op(ScalarMulKernelOp(rhs), self)
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, LhsTape: Tape<D>, Rhs> std::ops::Mul<Rhs>
    for Tensor<S, E, D, LhsTape>
where
    Self: TryMul<Rhs>,
{
    type Output = Self;
    fn mul(self, rhs: Rhs) -> Self::Output {
        self.try_mul(rhs).unwrap()
    }
}
#[cfg(test)]
mod tests {
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::build_test_device;

    #[test]
    fn test_mul_0d() {
        let dev = build_test_device!();
        let a = dev.tensor(2.0);
        let b = dev.tensor(3.0);

        let r = a.trace() * b.clone();
        assert_eq!(r.array(), 6.0);
        let g = r.backward();
        assert_eq!(g.get(&a).array(), 3.0);
        assert_eq!(g.get(&b).array(), 2.0);
    }

    #[test]
    fn test_mul_1d() {
        let dev = build_test_device!();
        let a = dev.tensor([1.0, 2.0, 3.0]);
        let b = dev.tensor([1.0, -1.0, 0.0]);

        let r = a.trace() * b.clone();
        assert_eq!(r.array(), [1.0, -2.0, 0.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&a).array(), [1.0 / 3.0, -1.0 / 3.0, 0.0]);
        assert_eq!(g.get(&b).array(), [1.0 / 3.0, 2.0 / 3.0, 1.0]);
    }

    #[test]
    fn test_mul_2d() {
        let dev = build_test_device!();
        let a = dev.tensor([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b = dev.tensor([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let r = a.trace() * b.clone();
        assert_eq!(
            r.array(),
            [
                [0.3415743, 0.06565552, 0.056385003],
                [0.46729425, 0.2581082, 0.03236696]
            ]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.get(&a).array(),
            [
                [0.08665001, 0.06406667, 0.06265],
                [0.13765001, 0.06136667, 0.006466667]
            ]
        );
        assert_eq!(
            g.get(&b).array(),
            [
                [0.109500006, 0.028466668, 0.025000002],
                [0.0943, 0.11683333, 0.13903335]
            ]
        );
    }

    #[test]
    fn test_scalar_mul_0d() {
        let dev = build_test_device!();
        let x = dev.tensor(1.0);
        let r = x.trace() * 0.5;
        assert_eq!(r.array(), 0.5);
        let g = r.exp().backward();
        assert_eq!(g.get(&x).array(), 0.8243606);
    }

    #[test]
    fn test_scalar_mul_1d() {
        let dev = build_test_device!();
        let x = dev.tensor([0.0, 1.0, 2.0]);
        let r = x.trace() * 0.5;
        assert_eq!(r.array(), [0.0, 0.5, 1.0]);
        let g = r.exp().sum().backward();
        assert_eq!(g.get(&x).array(), [0.5, 0.8243606, 1.3591409]);
    }

    #[test]
    fn test_scalar_mul_2d() {
        let dev = build_test_device!();
        let x = dev.tensor([[1.0; 2]; 3]);
        let r = x.trace() * 0.5;
        assert_eq!(r.array(), [[0.5; 2]; 3]);
        let g = r.exp().sum().backward();
        assert_eq!(g.get(&x).array(), [[0.8243606; 2]; 3]);
    }
}
