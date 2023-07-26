mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::*;
use crate::{
    shapes::*,
    tensor::{HasErr, Merge, Storage, Tape, Tensor},
};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct BinaryAddKernelOp;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarAddKernelOp<E> {
    scalar: E,
}

/// Element wise and scalar addition.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r = a + dev.ones();
/// assert_eq!(r.array(), [[2.0, 3.0, 4.0], [0.0, -1.0, -2.0]]);
/// ```
///
/// Adding a scalar:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r = a + 1.0;
/// assert_eq!(r.array(), [[2.0, 3.0, 4.0], [0.0, -1.0, -2.0]]);
/// ```
pub fn add<S: Shape, E: Dtype, D, T: Tape<E, D> + Merge<R>, R: Default>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, R>,
) -> Tensor<S, E, D, T>
where
    D: BinaryKernel<BinaryAddKernelOp, E>,
{
    lhs + rhs
}

/// Fallible version of [std::ops::Add]. See [add]
pub trait TryAdd<Rhs = Self>: HasErr {
    fn try_add(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D, LhsTape: Tape<E, D>, R> TryAdd<Tensor<S, E, D, R>>
    for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<BinaryAddKernelOp, E>,
    LhsTape: Merge<R>,
{
    /// See [add]
    fn try_add(self, rhs: Tensor<S, E, D, R>) -> Result<Self, Self::Err> {
        try_binary_op(BinaryAddKernelOp, self, rhs)
    }
}

impl<S: Shape, E: Dtype, D: UnaryKernel<ScalarAddKernelOp<E>, E>, T: Tape<E, D>> TryAdd<E>
    for Tensor<S, E, D, T>
{
    /// See [add]
    fn try_add(self, rhs: E) -> Result<Self, Self::Err> {
        try_unary_op(ScalarAddKernelOp { scalar: rhs }, self)
    }
}

#[cfg(feature = "f16")]
impl<S: Shape, D: UnaryKernel<ScalarAddKernelOp<half::f16>, half::f16>, T: Tape<half::f16, D>>
    TryAdd<f32> for Tensor<S, half::f16, D, T>
{
    /// See [add]
    fn try_add(self, rhs: f32) -> Result<Self, Self::Err> {
        let scalar = half::f16::from_f32(rhs);
        try_unary_op(ScalarAddKernelOp { scalar }, self)
    }
}

#[cfg(feature = "f16")]
impl<
        S: Shape,
        D: UnaryKernel<
            ScalarAddKernelOp<crate::dtypes::AMP<half::f16>>,
            crate::dtypes::AMP<half::f16>,
        >,
        T: Tape<crate::dtypes::AMP<half::f16>, D>,
    > TryAdd<f32> for Tensor<S, crate::dtypes::AMP<half::f16>, D, T>
{
    /// See [add]
    fn try_add(self, rhs: f32) -> Result<Self, Self::Err> {
        let scalar = crate::dtypes::AMP(half::f16::from_f32(rhs));
        try_unary_op(ScalarAddKernelOp { scalar }, self)
    }
}

impl<S: Shape, E: Dtype, D: Storage<E>, LhsTape: Tape<E, D>, Rhs> std::ops::Add<Rhs>
    for Tensor<S, E, D, LhsTape>
where
    Self: TryAdd<Rhs>,
{
    type Output = Self;
    /// See [add]
    fn add(self, rhs: Rhs) -> Self::Output {
        self.try_add(rhs).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{shapes::*, tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_add_0d() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor(1.0f64).to_dtype::<TestDtype>();
        let b = dev.tensor(1.0f64).to_dtype::<TestDtype>();

        let r = a.leaky_trace() + b.clone();
        assert_close_to_literal!(r, 2.0);
        let g = r.backward();
        assert_close_to_literal!(g.get(&a), 1.0);
        assert_close_to_literal!(g.get(&b), 1.0);
    }

    #[test]
    fn test_add_1d() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([1.0f64, 2.0, 3.0]).to_dtype::<TestDtype>();
        let b = dev.tensor([1.0f64, -1.0, 0.0]).to_dtype::<TestDtype>();

        let r = a.leaky_trace() + b.clone();
        assert_close_to_literal!(r, [2.0, 1.0, 3.0]);
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&a), [1.0 / 3.0; 3]);
        assert_close_to_literal!(g.get(&b), [1.0 / 3.0; 3]);
    }

    #[test]
    fn test_add_2d() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[0.6570f64, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]])
            .to_dtype::<TestDtype>();
        let b = dev
            .tensor([[0.5199f64, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]])
            .to_dtype::<TestDtype>();

        let r = a.leaky_trace() + b.clone();
        assert_close_to_literal!(r, [[1.1769, 0.5552, 0.5259], [1.3917, 1.0692, 0.873]]);
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&a), [[1.0 / 6.0; 3]; 2]);
        assert_close_to_literal!(g.get(&b), [[1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_add_broadcast_bottom() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[0.6570f64, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]])
            .to_dtype::<TestDtype>();
        let b = dev
            .tensor([[0.5199f64, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]])
            .to_dtype::<TestDtype>();

        let a2 = a.broadcast::<Rank3<2, 3, 4>, _>();
        let b2 = b.broadcast::<Rank3<2, 3, 4>, _>();

        let r = a2.leaky_trace() + b2.clone();
        assert_close_to_literal!(
            r,
            [
                [[1.1769; 4], [0.5552; 4], [0.5259; 4]],
                [[1.3917; 4], [1.0692; 4], [0.873; 4]]
            ]
        );
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&a2), [[[1.0 / 6.0; 4]; 3]; 2]);
        assert_close_to_literal!(g.get(&b2), [[[1.0 / 6.0; 4]; 3]; 2]);
    }

    #[test]
    fn test_add_broadcast_top() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[0.6570f64, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]])
            .to_dtype::<TestDtype>();
        let b = dev
            .tensor([[0.5199f64, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]])
            .to_dtype::<TestDtype>();

        let a2 = a.broadcast::<Rank3<4, 2, 3>, _>();
        let b2 = b.broadcast::<Rank3<4, 2, 3>, _>();

        let r = a2.leaky_trace() + b2.clone();
        assert_close_to_literal!(r, [[[1.1769, 0.5552, 0.5259], [1.3917, 1.0692, 0.873]]; 4]);
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&a2), [[[1.0 / 6.0; 3]; 2]; 4]);
        assert_close_to_literal!(g.get(&b2), [[[1.0 / 6.0; 3]; 2]; 4]);
    }

    #[test]
    fn test_scalar_add_0d() {
        let dev: TestDevice = Default::default();
        let x: Tensor<(), TestDtype, _> = dev.zeros();
        let r = x.leaky_trace() + 1.0;
        assert_close_to_literal!(r, 1.0);
        let g = r.exp().backward();
        assert_close_to_literal!(g.get(&x), f64::exp(1.0));
    }

    #[test]
    fn test_scalar_add_1d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([0.0, 1.0, 2.0]).to_dtype::<TestDtype>();
        let r = x.leaky_trace() + 0.5;
        assert_close_to_literal!(r, [0.5, 1.5, 2.5]);
        let g = r.exp().sum().backward();
        assert_close_to_literal!(g.get(&x), [1.6487212, 4.481689, 12.182494]);
    }

    #[test]
    fn test_scalar_add_2d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([[0.0; 2]; 3]).to_dtype::<TestDtype>();
        let r = x.leaky_trace() + 0.5;
        assert_close_to_literal!(r, [[0.5; 2]; 3]);
        let g = r.exp().sum().backward();
        assert_close_to_literal!(g.get(&x), [[1.6487212; 2]; 3]);
    }
}
