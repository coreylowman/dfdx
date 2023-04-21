mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::*;
use crate::{shapes::*, tensor::*};

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
pub fn mul<S: Shape, E: Dtype, D, T: Tape<E, D> + Merge<R>, R: Default>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, R>,
) -> Tensor<S, E, D, T>
where
    D: BinaryKernel<BinaryMulKernelOp, E>,
{
    lhs * rhs
}

/// Fallible version of [std::ops::Mul]. See [mul].
pub trait TryMul<Rhs = Self>: HasErr {
    fn try_mul(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: BinaryKernel<BinaryMulKernelOp, E>, LhsTape: Tape<E, D>, R>
    TryMul<Tensor<S, E, D, R>> for Tensor<S, E, D, LhsTape>
where
    LhsTape: Merge<R>,
{
    fn try_mul(self, rhs: Tensor<S, E, D, R>) -> Result<Self, Self::Err> {
        try_binary_op(BinaryMulKernelOp, self, rhs)
    }
}

impl<S: Shape, E: Dtype, D: UnaryKernel<ScalarMulKernelOp<E>, E>, T: Tape<E, D>> TryMul<E>
    for Tensor<S, E, D, T>
{
    fn try_mul(self, rhs: E) -> Result<Self, Self::Err> {
        try_unary_op(ScalarMulKernelOp { scalar: rhs }, self)
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage, LhsTape: Tape<E, D>, Rhs> std::ops::Mul<Rhs>
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
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_mul_0d() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor(2.0);
        let b: Tensor<_, TestDtype, _> = dev.tensor(3.0);

        let r = a.leaky_trace() * b.clone();
        assert_close_to_literal!(r, 6.0);
        let g = r.backward();
        assert_close_to_literal!(g.get(&a), 3.0);
        assert_close_to_literal!(g.get(&b), 2.0);
    }

    #[test]
    fn test_mul_1d() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([1.0, 2.0, 3.0]);
        let b: Tensor<_, TestDtype, _> = dev.tensor([1.0, -1.0, 0.0]);

        let r = a.leaky_trace() * b.clone();
        assert_close_to_literal!(r, [1.0, -2.0, 0.0]);
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&a), [1.0 / 3.0, -1.0 / 3.0, 0.0]);
        assert_close_to_literal!(g.get(&b), [1.0 / 3.0, 2.0 / 3.0, 1.0]);
    }

    #[test]
    fn test_mul_2d() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> =
            dev.tensor([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b: Tensor<_, TestDtype, _> =
            dev.tensor([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let r = a.leaky_trace() * b.clone();
        assert_close_to_literal!(
            r,
            [
                [0.3415743, 0.06565552, 0.056385003],
                [0.46729425, 0.2581082, 0.03236696],
            ]
        );
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&a),
            [
                [0.08665001, 0.06406667, 0.06265],
                [0.13765001, 0.06136667, 0.006466667],
            ]
        );
        assert_close_to_literal!(
            g.get(&b),
            [
                [0.109500006, 0.028466668, 0.025000002],
                [0.0943, 0.11683333, 0.13903335],
            ]
        );
    }

    #[test]
    fn test_scalar_mul_0d() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor(1.0);
        let r = x.leaky_trace() * 0.5;
        assert_close_to_literal!(r, 0.5);
        let g = r.exp().backward();
        assert_close_to_literal!(g.get(&x), 0.8243606);
    }

    #[test]
    fn test_scalar_mul_1d() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([0.0, 1.0, 2.0]);
        let r = x.leaky_trace() * 0.5;
        assert_close_to_literal!(r, [0.0, 0.5, 1.0]);
        let g = r.exp().sum().backward();
        assert_close_to_literal!(g.get(&x), [0.5, 0.8243606, 1.3591409]);
    }

    #[test]
    fn test_scalar_mul_2d() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([[1.0; 2]; 3]);
        let r = x.leaky_trace() * 0.5;
        assert_close_to_literal!(r, [[0.5; 2]; 3]);
        let g = r.exp().sum().backward();
        assert_close_to_literal!(g.get(&x), [[0.8243606; 2]; 3]);
    }
}
