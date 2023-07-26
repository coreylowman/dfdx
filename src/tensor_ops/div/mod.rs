mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::*;
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarDivKernelOp<E> {
    pub(crate) scalar: E,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct BinaryDivKernelOp;

/// Element wise and scalar division.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = dev.tensor([[1.0, 0.5, 1.0], [0.5, 1.0, 3.0]]);
/// let r = a / b;
/// assert_eq!(r.array(), [[1.0, 4.0, 3.0], [-2.0, -2.0, -1.0]]);
/// ```
///
/// Scalar example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r = a / 2.0;
/// assert_eq!(r.array(), [[0.5, 1.0, 1.5], [-0.5, -1.0, -1.5]]);
/// ```
pub fn div<S: Shape, E: Dtype, D, T: Tape<E, D> + Merge<R>, R: Default>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, R>,
) -> Tensor<S, E, D, T>
where
    D: BinaryKernel<BinaryDivKernelOp, E>,
{
    lhs / rhs
}

/// Fallible version of [std::ops::Div]. See [div]
pub trait TryDiv<Rhs = Self>: HasErr {
    fn try_div(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D, LhsTape: Tape<E, D>, R> TryDiv<Tensor<S, E, D, R>>
    for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<BinaryDivKernelOp, E>,
    LhsTape: Merge<R>,
{
    /// See [div]
    fn try_div(self, rhs: Tensor<S, E, D, R>) -> Result<Self, Self::Err> {
        try_binary_op(BinaryDivKernelOp, self, rhs)
    }
}

impl<S: Shape, E: Dtype, D: UnaryKernel<ScalarDivKernelOp<E>, E>, T: Tape<E, D>> TryDiv<E>
    for Tensor<S, E, D, T>
{
    /// See [div]
    fn try_div(self, rhs: E) -> Result<Self, Self::Err> {
        try_unary_op(ScalarDivKernelOp { scalar: rhs }, self)
    }
}

#[cfg(feature = "f16")]
impl<S: Shape, D: UnaryKernel<ScalarDivKernelOp<half::f16>, half::f16>, T: Tape<half::f16, D>>
    TryDiv<f32> for Tensor<S, half::f16, D, T>
{
    /// See [div]
    fn try_div(self, rhs: f32) -> Result<Self, Self::Err> {
        let scalar = half::f16::from_f32(rhs);
        try_unary_op(ScalarDivKernelOp { scalar }, self)
    }
}

#[cfg(feature = "f16")]
impl<
        S: Shape,
        D: UnaryKernel<
            ScalarDivKernelOp<crate::dtypes::AMP<half::f16>>,
            crate::dtypes::AMP<half::f16>,
        >,
        T: Tape<crate::dtypes::AMP<half::f16>, D>,
    > TryDiv<f32> for Tensor<S, crate::dtypes::AMP<half::f16>, D, T>
{
    /// See [div]
    fn try_div(self, rhs: f32) -> Result<Self, Self::Err> {
        let scalar = crate::dtypes::AMP(half::f16::from_f32(rhs));
        try_unary_op(ScalarDivKernelOp { scalar }, self)
    }
}

impl<S: Shape, E: Dtype, D: Storage<E>, LhsTape: Tape<E, D>, Rhs> std::ops::Div<Rhs>
    for Tensor<S, E, D, LhsTape>
where
    Self: TryDiv<Rhs>,
{
    type Output = Self;
    /// See [div]
    fn div(self, rhs: Rhs) -> Self::Output {
        self.try_div(rhs).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::*;

    #[test]
    fn test_div_0d() {
        let dev: TestDevice = Default::default();

        let a = dev.tensor(2.0).to_dtype::<TestDtype>();
        let b = dev.tensor(4.0).to_dtype::<TestDtype>();

        let r = b.leaky_trace() / a.clone();
        assert_close_to_literal!(r, 2.0);
        let g = r.backward();
        assert_close_to_literal!(g.get(&a), -1.0);
        assert_close_to_literal!(g.get(&b), 0.5);
    }

    #[test]
    fn test_div_1d() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([1.0, 2.0, 3.0]).to_dtype::<TestDtype>();
        let b = dev.tensor([1.0, -1.0, 0.0]).to_dtype::<TestDtype>();

        let r = b.leaky_trace() / a.clone();
        assert_close_to_literal!(r, [1.0, -0.5, 0.0]);
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&a), [-1.0 / 3.0, 1.0 / 12.0, 0.0]);
        assert_close_to_literal!(g.get(&b), [1.0 / 3.0, 1.0 / 6.0, 0.11111112]);
    }

    #[test]
    fn test_div_2d() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]])
            .to_dtype::<TestDtype>();
        let b = dev
            .tensor([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]])
            .to_dtype::<TestDtype>();

        let r = b.leaky_trace() / a.clone();
        assert_close_to_literal!(
            r,
            [
                [0.79132426, 2.2505856, 2.5059998],
                [1.4597031, 0.52524966, 0.046511628],
            ]
        );
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&a),
            [
                [-0.20074183, -2.19612169, -2.78444433],
                [-0.42998207, -0.12488105, -0.00929266]
            ]
        );
        assert_close_to_literal!(
            g.get(&b),
            &[
                [0.25367835, 0.97580016, 1.11111104],
                [0.29456815, 0.23775560, 0.19979222]
            ]
        );
    }

    #[test]
    fn test_scalar_div_0d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor(1.0).to_dtype::<TestDtype>();
        let r = x.leaky_trace() / 2.0;
        assert_close_to_literal!(r, 0.5);
        let g = r.exp().backward();
        assert_close_to_literal!(g.get(&x), 0.8243606);
    }

    #[test]
    fn test_scalar_div_1d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([0.0, 1.0, 2.0]).to_dtype::<TestDtype>();
        let r = x.leaky_trace() / 2.0;
        assert_close_to_literal!(r, [0.0, 0.5, 1.0]);
        let g = r.exp().sum().backward();
        assert_close_to_literal!(g.get(&x), [0.5, 0.8243606, 1.3591409]);
    }

    #[test]
    fn test_scalar_div_2d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([[1.0; 2]; 3]).to_dtype::<TestDtype>();
        let r = x.leaky_trace() / 2.0;
        assert_close_to_literal!(r, [[0.5; 2]; 3]);
        let g = r.exp().sum().backward();
        assert_close_to_literal!(g.get(&x), [[0.8243606; 2]; 3]);
    }
}
