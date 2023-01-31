mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::*;
use crate::{gradients::*, shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct BinaryMulKernelOp;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarMulKernelOp<E> {
    scalar: E,
}

/// Element wise and scalar multiplication.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r = a * dev.ones();
/// assert_eq!(r.array(), [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// ```
///
/// Scalar example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r = a * 2.0;
/// assert_eq!(r.array(), [[2.0, 4.0, 6.0], [-2.0, -4.0, -6.0]]);
/// ```
pub fn mul<S: ConstShape, E: Dtype, D, T: Tape<D> + Merge<RhsTape>, RhsTape: Tape<D>>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, RhsTape>,
) -> Tensor<S, E, D, T>
where
    D: BinaryKernel<BinaryMulKernelOp, E>,
{
    lhs * rhs
}

/// Fallible version of std::ops::Mul with compile time known shapes
pub trait TryConstMul<Rhs = Self>: HasErr {
    fn try_mul(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

/// Runtime shape checked multiplication
pub trait TryCheckedMul<Rhs = Self>: HasErr {
    fn checked_mul(self, rhs: Rhs) -> Option<Self> {
        self.try_checked_mul(rhs).map(Result::unwrap)
    }
    fn try_checked_mul(self, rhs: Rhs) -> Option<Result<Self, Self::Err>>;
}

impl<S: ConstShape, E: Dtype, D, LhsTape: Tape<D>, RhsTape: Tape<D>>
    TryConstMul<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<BinaryMulKernelOp, E>,
    LhsTape: Merge<RhsTape>,
{
    fn try_mul(self, rhs: Tensor<S, E, D, RhsTape>) -> Result<Self, Self::Err> {
        try_binary_op(BinaryMulKernelOp, self, rhs)
    }
}

impl<S: Shape, E: Dtype, D, LhsTape: Tape<D>, RhsTape: Tape<D>>
    TryCheckedMul<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<BinaryMulKernelOp, E>,
    LhsTape: Merge<RhsTape>,
{
    fn try_checked_mul(self, rhs: Tensor<S, E, D, RhsTape>) -> Option<Result<Self, Self::Err>> {
        try_checked_binary_op(BinaryMulKernelOp, self, rhs)
    }
}

impl<S: Shape, E: Dtype, D: UnaryKernel<ScalarMulKernelOp<E>, E>, T: Tape<D>> TryConstMul<E>
    for Tensor<S, E, D, T>
{
    fn try_mul(self, rhs: E) -> Result<Self, Self::Err> {
        try_unary_op(ScalarMulKernelOp { scalar: rhs }, self)
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage, LhsTape: Tape<D>, Rhs> std::ops::Mul<Rhs>
    for Tensor<S, E, D, LhsTape>
where
    Self: TryConstMul<Rhs>,
{
    type Output = Self;
    fn mul(self, rhs: Rhs) -> Self::Output {
        self.try_mul(rhs).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_mul_0d() {
        let dev: TestDevice = Default::default();
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
        let dev: TestDevice = Default::default();
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
        let dev: TestDevice = Default::default();
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
        let dev: TestDevice = Default::default();
        let x = dev.tensor(1.0);
        let r = x.trace() * 0.5;
        assert_eq!(r.array(), 0.5);
        let g = r.exp().backward();
        assert_eq!(g.get(&x).array(), 0.8243606);
    }

    #[test]
    fn test_scalar_mul_1d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([0.0, 1.0, 2.0]);
        let r = x.trace() * 0.5;
        assert_eq!(r.array(), [0.0, 0.5, 1.0]);
        let g = r.exp().sum().backward();
        assert_eq!(g.get(&x).array(), [0.5, 0.8243606, 1.3591409]);
    }

    #[test]
    fn test_scalar_mul_2d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([[1.0; 2]; 3]);
        let r = x.trace() * 0.5;
        assert_eq!(r.array(), [[0.5; 2]; 3]);
        let g = r.exp().sum().backward();
        assert_eq!(g.get(&x).array(), [[0.8243606; 2]; 3]);
    }
}
