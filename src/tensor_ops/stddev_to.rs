use super::*;
use crate::{gradients::Tape, shapes::*, tensor::*};

/// Reduces `Axes` of `T` by computing std deviation of all values in those axes.
/// Result [Tensor] has smaller number of dimensions.
///
/// **Pytorch equivalent**: `t.std(Axes, unbiased=False)`
///
/// **Related functions**: [var()], [sqrt()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor2D<2, 3> = tensor([[2.0, 3.0, 4.0], [3.0, 6.0, 9.0]]);
/// let r: Tensor1D<2> = t.stddev(0.0);
/// assert_eq!(r.data(), &[0.6666667_f32.sqrt(), 6.0_f32.sqrt()]);
/// ```
pub trait StddevTo: HasErr + HasShape {
    fn stddev<Dst: Shape, Ax: Axes>(self, epsilon: f32) -> Self::WithShape<Dst>
    where
        Self::Shape: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        self.try_stddev(epsilon).unwrap()
    }
    fn try_stddev<Dst: Shape, Ax: Axes>(
        self,
        epsilon: f32,
    ) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>;
}

impl<S: Shape, D: Device<f32>, T: Tape<D>> StddevTo for Tensor<S, f32, D, T> {
    fn try_stddev<Dst: Shape, Ax: Axes>(
        self,
        epsilon: f32,
    ) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        self.try_var()?.try_add(epsilon)?.try_sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::build_test_device;

    #[test]
    fn test_std_axis_0_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r = t.trace().stddev::<Rank1<4>, _>(1e-8);
        assert_eq!(r.array(), [0.5, 0.0001, 1.0, 3.0]);
        let g = r.mean().backward();
        assert_eq!(
            g.get(&t).array(),
            [[0.125, 0.0, -0.125, -0.125], [-0.125, 0.0, 0.125, 0.125]]
        );
    }

    #[test]
    fn test_std_axis_1_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r = t.trace().stddev::<Rank1<2>, _>(0.0);
        assert_eq!(r.array(), [1.118034, 3.7666297]);
        let g = r.mean().backward();
        assert_eq!(
            g.get(&t).array(),
            [
                [-0.16770509, -0.0559017, 0.0559017, 0.16770509],
                [-0.14104122, -0.07466887, 0.024889633, 0.19082046]
            ]
        );
    }
}
