use crate::{
    arrays::{Dtype, HasShape, Shape},
    devices::{Device, HasErr},
    gradients::Tape,
    tensor::Tensor,
};

use super::*;

/// Reduces `Axes` of `T` by computing variance of all values in those axes.
/// Result [Tensor] has smaller number of dimensions.
///
/// **Pytorch equivalent**: `t.var(Axes, unbiased=False)`
///
/// **Related functions**: [stddev()], [mean()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor2D<2, 3> = tensor([[2.0, 3.0, 4.0], [3.0, 6.0, 9.0]]);
/// let r: Tensor1D<2> = t.var();
/// assert_eq!(r.data(), &[0.6666667, 6.0]);
/// ```
pub trait VarTo<T, Axes>: HasErr {
    fn var(self) -> T {
        self.try_var().unwrap()
    }
    fn try_var(self) -> Result<T, Self::Err>;
}

impl<Src: Shape, Dst: Shape, Axes, E: Dtype, D: Device, T: Tape<D>>
    VarTo<Tensor<Dst, E, D, T>, Axes> for Tensor<Src, E, D, T>
where
    Self: MeanTo<Tensor<Dst, E, D, T>, Axes, Err = D::Err>
        + TrySub<Self>
        + TrySquare
        + SumTo<Tensor<Dst, E, D, T>, Axes>,
    Tensor<Dst, E, D, T>: BroadcastTo<Self, Axes, Err = D::Err> + TryDiv<f32, Err = D::Err>,
{
    fn try_var(self) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        let mean = self
            .with_empty_tape()
            .try_mean()?
            .try_broadcast_to(self.shape())?;
        mean.try_sub(self)?.try_square()?.try_mean()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::AsArray;
    use crate::tensor::*;
    use crate::tests::build_test_device;

    #[test]
    fn test_var_axis_0_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r: Tensor1D<4, _, _> = t.trace().var();
        assert_eq!(r.as_array(), [0.25, 0.0, 1.0, 9.0]);
        let g = r.mean().backward();
        assert_eq!(
            g.get(&t).as_array(),
            [[0.125, 0.0, -0.25, -0.75], [-0.125, 0.0, 0.25, 0.75]]
        );
    }

    #[test]
    fn test_var_axis_1_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r: Tensor1D<2, _, _> = t.trace().var();
        assert_eq!(r.as_array(), [1.25, 14.1875]);
        let g = r.mean().backward();
        assert_eq!(
            g.get(&t).as_array(),
            [
                [-0.375, -0.125, 0.125, 0.375],
                [-1.0625, -0.5625, 0.1875, 1.4375]
            ]
        );
    }
}
