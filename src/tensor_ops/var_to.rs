use super::*;
use crate::{shapes::*, tensor::*};

/// Reduction alogn multiple axes using variance
pub trait VarTo: HasErr + HasShape {
    /// Result [Tensor] has smaller number of dimensions.
    ///
    /// **Pytorch equivalent**: `t.var(Axes, unbiased=False)`
    ///
    /// Examples:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let t = dev.tensor([[2.0f32, 3.0, 4.0], [3.0, 6.0, 9.0]]);
    /// let r = t.var::<Rank1<2>, _>(); // or `var::<_, Axis<1>>()`
    /// assert_eq!(r.array(), [0.6666667, 6.0]);
    /// ```
    fn var<Dst: Shape, Ax: Axes>(self) -> Self::WithShape<Dst>
    where
        Self::Shape: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        self.try_var().unwrap()
    }
    /// Fallible version of [VarTo::var]
    fn try_var<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>;
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> VarTo for Tensor<S, E, D, T> {
    fn try_var<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let mean = self
            .retaped::<T>()
            .try_mean::<Dst, Ax>()?
            .try_broadcast_like(self.shape())?;
        mean.try_sub(self)?.try_square()?.try_mean()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_var_axis_0_2d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r = t.leaky_trace().var::<Rank1<4>, _>();
        assert_close_to_literal!(r, [0.25, 0.0, 1.0, 9.0]);
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&t),
            [[0.125, 0.0, -0.25, -0.75], [-0.125, 0.0, 0.25, 0.75]]
        );
    }

    #[test]
    fn test_var_axis_1_2d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r = t.leaky_trace().var::<Rank1<2>, _>();
        assert_close_to_literal!(r, [1.25, 14.1875]);
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&t),
            [
                [-0.375, -0.125, 0.125, 0.375],
                [-1.0625, -0.5625, 0.1875, 1.4375]
            ]
        );
    }
}
