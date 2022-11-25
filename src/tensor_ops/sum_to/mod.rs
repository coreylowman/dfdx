mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{Device, HasErr},
    gradients::Tape,
    tensor::Tensor,
};

use super::ops::{try_unary_op, UnaryKernel};

/// Sum values along axes `Axes` of `T`.
///
/// **Pytorch equivalent**: `t.sum(Axes)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor3D<2, 3, 4> = TensorCreator::zeros();
/// let _: Tensor0D = t.sum();
/// ```
///
/// Reducing 1 axis:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor3D<2, 3, 4> = TensorCreator::zeros();
/// let _: Tensor2D<3, 4> = t.sum();
/// ```
///
/// Reducing multiple axes:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor3D<2, 3, 4> = TensorCreator::zeros();
/// let _: Tensor1D<4> = t.sum();
/// ```
pub trait SumTo<T, Axes>: HasErr {
    fn sum(self) -> T {
        self.try_sum().unwrap()
    }
    fn try_sum(self) -> Result<T, Self::Err>;
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct SumKernelOp<Axes>(std::marker::PhantomData<Axes>);

impl<Src: Shape, Dst: Shape, Axes: 'static + Copy + Default, E: Dtype, D: Device, T: Tape<D>>
    SumTo<Tensor<Dst, E, D, T>, Axes> for Tensor<Src, E, D, T>
where
    D: UnaryKernel<SumKernelOp<Axes>, Src, Dst, E>,
{
    fn try_sum(self) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        try_unary_op(Default::default(), self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::AsArray;
    use crate::devices::Randn;
    use crate::gradients::OwnedTape;
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_sum_1d() {
        let dev = build_test_device!();
        let t = dev.tensor([1.0, 2.0, 3.0]);
        let r: Tensor0D<_, OwnedTape<_>> = t.trace().sum();
        assert_eq!(r.as_array(), 6.0);
        // NOTE: .exp() to make sure its using result grad properly
        let g = r.exp().backward();
        assert_eq!(g.get(&t).as_array(), [403.4288; 3]);
    }

    #[test]
    fn test_sum_axis_0_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r: Tensor1D<3, _, _> = t.trace().sum();
        assert_eq!(r.as_array(), [-1.0, 6.0, -3.0]);
        let g = r.exp().mean().backward();
        assert_eq!(
            g.get(&t).as_array(),
            [[0.12262648, 134.47627, 0.01659569]; 2]
        );
    }

    #[test]
    fn test_sum_axis_1_2d() {
        let dev = build_test_device!();
        let t: Tensor2D<2, 3, _> = dev.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r: Tensor1D<2, _, _> = t.trace().sum();
        assert_eq!(r.as_array(), [6.0, -4.0]);
        let g = r.exp().mean().backward();
        assert_eq!(g.get(&t).as_array(), [[201.7144; 3], [0.00915782; 3]]);
    }

    #[test]
    fn test_sum_axes_3d_to_1d() {
        let dev = build_test_device!();
        let t: Tensor3D<2, 3, 4, _, _> = dev.randn();
        let r: Tensor1D<3, _, _> = t.trace().sum();
        let a: Tensor2D<3, 4, _, _> = t.trace().sum();
        let r2: Tensor1D<3, _, _> = a.sum();
        assert_close(&r.as_array(), &r2.as_array());
        let g = r.mean().backward();
        let g2 = r2.mean().backward();
        assert_close(&g.get(&t).as_array(), &g2.get(&t).as_array());
    }
}
