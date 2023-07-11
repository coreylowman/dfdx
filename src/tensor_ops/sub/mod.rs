mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::*;
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct BinarySubKernelOp;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarSubKernelOp<E> {
    scalar: E,
}

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
pub fn sub<S: Shape, E: Dtype, D, T: Tape<E, D> + Merge<R>, R>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, R>,
) -> Tensor<S, E, D, T>
where
    D: BinaryKernel<BinarySubKernelOp, E>,
{
    lhs - rhs
}

/// Fallible version of [std::ops::Sub]. See [sub]
pub trait TrySub<Rhs = Self>: HasErr {
    fn try_sub(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: BinaryKernel<BinarySubKernelOp, E>, LTape: Tape<E, D>, R>
    TrySub<Tensor<S, E, D, R>> for Tensor<S, E, D, LTape>
where
    LTape: Merge<R>,
{
    fn try_sub(self, rhs: Tensor<S, E, D, R>) -> Result<Self, Self::Err> {
        try_binary_op(BinarySubKernelOp, self, rhs)
    }
}

impl<S: Shape, E: Dtype, D: UnaryKernel<ScalarSubKernelOp<E>, E>, T: Tape<E, D>> TrySub<E>
    for Tensor<S, E, D, T>
{
    fn try_sub(self, rhs: E) -> Result<Self, Self::Err> {
        try_unary_op(ScalarSubKernelOp { scalar: rhs }, self)
    }
}

#[cfg(feature = "f16")]
impl<S: Shape, D: UnaryKernel<ScalarSubKernelOp<half::f16>, half::f16>, T: Tape<half::f16, D>>
    TrySub<f32> for Tensor<S, half::f16, D, T>
{
    fn try_sub(self, rhs: f32) -> Result<Self, Self::Err> {
        let scalar = half::f16::from_f32(rhs);
        try_unary_op(ScalarSubKernelOp { scalar }, self)
    }
}

#[cfg(feature = "f16")]
impl<
        S: Shape,
        D: UnaryKernel<
            ScalarSubKernelOp<crate::dtypes::AMP<half::f16>>,
            crate::dtypes::AMP<half::f16>,
        >,
        T: Tape<crate::dtypes::AMP<half::f16>, D>,
    > TrySub<f32> for Tensor<S, crate::dtypes::AMP<half::f16>, D, T>
{
    fn try_sub(self, rhs: f32) -> Result<Self, Self::Err> {
        let scalar = crate::dtypes::AMP(half::f16::from_f32(rhs));
        try_unary_op(ScalarSubKernelOp { scalar }, self)
    }
}

impl<S: Shape, E: Dtype, D: Storage<E>, LTape: Tape<E, D>, Rhs> std::ops::Sub<Rhs>
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
    use crate::tests::*;

    #[test]
    fn test_sub_0d() {
        let dev: TestDevice = Default::default();

        let a = dev.tensor(1.0).to_dtype::<TestDtype>();
        let b = dev.tensor(1.0).to_dtype::<TestDtype>();

        let r = b.leaky_trace() - a.clone();
        assert_close_to_literal!(r, 0.0);
        let g = r.backward();
        assert_close_to_literal!(g.get(&a), -1.0);
        assert_close_to_literal!(g.get(&b), 1.0);
    }

    #[test]
    fn test_sub_1d() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([1.0, 2.0, 3.0]).to_dtype::<TestDtype>();
        let b = dev.tensor([1.0, -1.0, 0.0]).to_dtype::<TestDtype>();

        let r = b.leaky_trace() - a.clone();
        assert_close_to_literal!(r, [0.0, -3.0, -3.0]);
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&a), [-1.0 / 3.0; 3]);
        assert_close_to_literal!(g.get(&b), [1.0 / 3.0; 3]);
    }

    #[test]
    fn test_sub_2d() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]])
            .to_dtype::<TestDtype>();
        let b = dev
            .tensor([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]])
            .to_dtype::<TestDtype>();

        let r = b.leaky_trace() - a.clone();
        assert_close_to_literal!(r, [[-0.1371, 0.2136, 0.2259], [0.2601, -0.3328, -0.7954]]);
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&a), [[-1.0 / 6.0; 3]; 2]);
        assert_close_to_literal!(g.get(&b), [[1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_scalar_sub_0d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor(0.0).to_dtype::<TestDtype>();
        let r = x.leaky_trace() - 1.0;
        assert_close_to_literal!(r, -1.0);
        let g = r.exp().backward();
        assert_close_to_literal!(g.get(&x), f64::exp(-1.0));
    }

    #[test]
    fn test_scalar_sub_1d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([0.0, 1.0, 2.0]).to_dtype::<TestDtype>();
        let r = x.leaky_trace() - 1.0;
        assert_close_to_literal!(r, [-1.0, 0.0, 1.0]);
        let g = r.exp().sum().backward();
        assert_close_to_literal!(g.get(&x), [0.36787945, 1.0, 2.7182817]);
    }

    #[test]
    fn test_scalar_sub_2d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([[0.0; 2]; 3]).to_dtype::<TestDtype>();
        let r = x.leaky_trace() - 1.0;
        assert_close_to_literal!(r, [[-1.0; 2]; 3]);
        let g = r.exp().sum().backward();
        assert_close_to_literal!(g.get(&x), [[0.36787945; 2]; 3]);
    }
}
