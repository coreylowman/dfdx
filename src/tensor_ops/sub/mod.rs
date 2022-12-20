mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::{ops::*, Device};
use crate::{gradients::*, shapes::*, tensor::*};

#[derive(Debug, Default, Clone, Copy)]
pub struct BinarySubKernelOp;

#[derive(Debug, Clone, Copy)]
pub struct ScalarSubKernelOp<E>(E);

/// Element wise and scalar subtraction.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = dev.ones();
/// let r = a - b;
/// assert_eq!(r.array(), [[0.0, 1.0, 2.0], [-2.0, -3.0, -4.0]]);
/// ```
///
/// Scalar Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r = a - 1.0;
/// assert_eq!(r.array(), [[0.0, 1.0, 2.0], [-2.0, -3.0, -4.0]]);
/// ```
pub fn sub<S: Shape, E: Dtype, D: Device<E>, T: Tape<D> + Merge<RhsTape>, RhsTape: Tape<D>>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, RhsTape>,
) -> Tensor<S, E, D, T> {
    lhs - rhs
}

/// Fallible version of std::ops::Sub
pub trait TrySub<Rhs = Self>: HasErr {
    fn try_sub(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: Device<E>, LTape: Tape<D>, RTape: Tape<D>>
    TrySub<Tensor<S, E, D, RTape>> for Tensor<S, E, D, LTape>
where
    LTape: Merge<RTape>,
{
    fn try_sub(self, rhs: Tensor<S, E, D, RTape>) -> Result<Self, Self::Err> {
        try_binary_op(BinarySubKernelOp, self, rhs)
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> TrySub<E> for Tensor<S, E, D, T> {
    fn try_sub(self, rhs: E) -> Result<Self, Self::Err> {
        try_unary_op(ScalarSubKernelOp(rhs), self)
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, LTape: Tape<D>, Rhs> std::ops::Sub<Rhs>
    for Tensor<S, E, D, LTape>
where
    Self: TrySub<Rhs>,
{
    type Output = Self;
    fn sub(self, rhs: Rhs) -> Self::Output {
        self.try_sub(rhs).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::TestDevice;

    #[test]
    fn test_sub_0d() {
        let dev: TestDevice = Default::default();

        let a = dev.tensor(1.0);
        let b = dev.tensor(1.0);

        let r = b.trace() - a.clone();
        assert_eq!(r.array(), 0.0);
        let g = r.backward();
        assert_eq!(g.get(&a).array(), -1.0);
        assert_eq!(g.get(&b).array(), 1.0);
    }

    #[test]
    fn test_sub_1d() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([1.0, 2.0, 3.0]);
        let b = dev.tensor([1.0, -1.0, 0.0]);

        let r = b.trace() - a.clone();
        assert_eq!(r.array(), [0.0, -3.0, -3.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&a).array(), [-1.0 / 3.0; 3]);
        assert_eq!(g.get(&b).array(), [1.0 / 3.0; 3]);
    }

    #[test]
    fn test_sub_2d() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b = dev.tensor([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let r = b.trace() - a.clone();
        assert_eq!(
            r.array(),
            [
                [-0.13709998, 0.21360001, 0.2259],
                [0.2601, -0.33279997, -0.7954]
            ]
        );
        let g = r.mean().backward();
        assert_eq!(g.get(&a).array(), [[-1.0 / 6.0; 3]; 2]);
        assert_eq!(g.get(&b).array(), [[1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_scalar_sub_0d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor(0.0);
        let r = x.trace() - 1.0;
        assert_eq!(r.array(), -1.0);
        let g = r.exp().backward();
        assert_eq!(g.get(&x).array(), (-1.0f32).exp());
    }

    #[test]
    fn test_scalar_sub_1d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([0.0, 1.0, 2.0]);
        let r = x.trace() - 1.0;
        assert_eq!(r.array(), [-1.0, 0.0, 1.0]);
        let g = r.exp().sum().backward();
        assert_eq!(g.get(&x).array(), [0.36787945, 1.0, 2.7182817]);
    }

    #[test]
    fn test_scalar_sub_2d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([[0.0; 2]; 3]);
        let r = x.trace() - 1.0;
        assert_eq!(r.array(), [[-1.0; 2]; 3]);
        let g = r.exp().sum().backward();
        assert_eq!(g.get(&x).array(), [[0.36787945; 2]; 3]);
    }
}
