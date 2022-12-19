use super::*;
use crate::{gradients::Tape, shapes::*, tensor::*};

/// Reduction along multiple axes using `mean`.
pub trait MeanTo: HasErr + HasShape {
    /// Mean reduction. **Pytorch equivalent**: `t.mean(Axes)`
    ///
    /// Example:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let t = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let r = t.mean::<Rank0, _>(); // or `mean::<_, Axes2<0, 1>>()`
    /// assert_eq!(r.array(), 3.5);
    /// ```
    ///
    /// Reducing 1 axis:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let t = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let r = t.mean::<Rank1<2>, _>(); // or `mean::<_, Axis<1>>()`
    /// assert_eq!(r.array(), [2.0, 5.0]);
    /// ```
    fn mean<Dst: Shape, Ax: Axes>(self) -> Self::WithShape<Dst>
    where
        Self::Shape: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        self.try_mean().unwrap()
    }
    /// Fallible version of [MeanTo::mean]
    fn try_mean<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>;
}

impl<S: Shape, D: Device<f32>, T: Tape<D>> MeanTo for Tensor<S, f32, D, T> {
    fn try_mean<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let num_elements_reduced = <S as HasAxes<Ax>>::size(self.shape()) as f32;
        self.try_sum()?.try_div(num_elements_reduced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{assert_close, TestDevice};

    #[test]
    fn test_valids_mean_axis() {
        let dev: TestDevice = Default::default();
        let _ = dev.zeros::<Rank1<5>>().mean::<Rank0, _>();
        let _ = dev.zeros::<Rank2<5, 3>>().mean::<Rank1<3>, _>();
        let _ = dev.zeros::<Rank2<5, 3>>().mean::<Rank1<5>, _>();
        let _ = dev.zeros::<Rank3<7, 5, 3>>().mean::<Rank2<5, 3>, _>();
        let _ = dev.zeros::<Rank3<7, 5, 3>>().mean::<Rank2<7, 3>, _>();
        let _ = dev.zeros::<Rank3<7, 5, 3>>().mean::<Rank2<7, 5>, _>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().mean::<Rank3<7, 5, 3>, _>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().mean::<Rank3<9, 5, 3>, _>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().mean::<Rank3<9, 7, 3>, _>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().mean::<Rank3<9, 7, 5>, _>();
    }

    #[test]
    fn test_mean_1d() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([1.0, 2.0, 3.0]);
        let r = t.trace().mean();
        assert_eq!(r.array(), 2.0);
        // NOTE: .exp() so we cover the case where .mean() has to use result grad.
        let g = r.exp().backward();
        assert_eq!(g.get(&t).array(), [2.463019; 3]);
    }

    #[test]
    fn test_mean_2d() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let r = t.trace().mean();
        assert_eq!(r.array(), 3.5);
        let g = r.backward();
        assert_eq!(g.get(&t).array(), [[1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_mean_3d() {
        let dev: TestDevice = Default::default();
        let t = dev.ones::<Rank3<4, 2, 3>>();
        let r = t.trace().mean();
        assert_eq!(r.array(), 1.0);
        let g = r.backward();
        assert_eq!(g.get(&t).array(), [[[1.0 / 24.0; 3]; 2]; 4]);
    }

    #[test]
    fn test_mean_axis_0_2d() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r = t.trace().mean::<Rank1<3>, _>();
        assert_eq!(r.array(), [-0.5, 3.0, -1.5]);
        let g = r.exp().mean().backward();
        assert_eq!(g.get(&t).array(), [[0.10108845, 3.3475895, 0.037188362]; 2]);
    }

    #[test]
    fn test_mean_axis_1_2d() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r = t.trace().mean::<Rank1<2>, _>();
        assert_eq!(r.array(), [2.0, -4.0 / 3.0]);
        let g = r.exp().mean().backward();
        assert_eq!(g.get(&t).array(), [[1.2315094; 3], [0.043932855; 3]]);
    }

    #[test]
    fn test_mean_axes_3d_to_1d_02() {
        let dev: TestDevice = Default::default();
        let t = dev.randn::<Rank3<2, 3, 4>>();
        let r = t.trace().mean::<Rank1<3>, _>();
        let r2 = t.trace().sum::<_, Axis<0>>().sum::<_, Axis<1>>() / 8.0;
        assert_close(&r.array(), &r2.array());
        let g = r.mean().backward();
        let g2 = r2.mean().backward();
        assert_close(&g.get(&t).array(), &[[[1. / 24.; 4]; 3]; 2]);
        assert_close(&g.get(&t).array(), &g2.get(&t).array());
    }

    #[test]
    fn test_mean_axes_3d_to_1d_01() {
        let dev: TestDevice = Default::default();
        let t = dev.randn::<Rank3<2, 3, 4>>();
        let r = t.trace().mean::<Rank1<4>, _>();
        let r2 = t.sum::<_, Axis<0>>().sum::<_, Axis<0>>() / 6.0;
        assert_close(&r.array(), &r2.array());
    }
}
