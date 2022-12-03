use super::*;
use crate::{arrays::*, gradients::Tape, tensor::*};

/// Average the values along `Axes` of `T`.
///
/// **Pytorch equivalent**: `t.mean(Axes)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor2D<2, 3> = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let r: Tensor0D = t.mean();
/// assert_eq!(r.as_array(), 3.5);
/// ```
///
/// Reducing 1 axis:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor2D<2, 3> = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let r: Tensor1D<2> = t.mean();
/// assert_eq!(r.as_array(), [2.0, 5.0]);
/// ```
///
/// Specifying axes instead of output type:
/// ```rust
/// todo!()
/// ```
pub trait MeanInto<T, Axes>: HasErr {
    fn mean(self) -> T {
        self.try_mean().unwrap()
    }
    fn try_mean(self) -> Result<T, Self::Err>;
}

impl<Ax: Axes, Src: Shape + HasAxes<Ax>, Dst: Shape, D: Device<f32>, T: Tape<D>>
    MeanInto<Tensor<Dst, f32, D, T>, Ax> for Tensor<Src, f32, D, T>
where
    Src: ReduceShapeTo<Dst, Ax>,
{
    fn try_mean(self) -> Result<Tensor<Dst, f32, D, T>, Self::Err> {
        let num_elements_reduced = <Src as HasAxes<Ax>>::size(self.shape()) as f32;
        self.try_sum()?.try_div(num_elements_reduced)
    }
}

pub trait MeanTo<Ax: Axes>: HasShape + HasErr {
    fn mean_to<Dst: Shape>(self) -> Self::With<Dst>
    where
        Self: MeanInto<Self::With<Dst>, Ax>,
    {
        self.mean()
    }

    fn try_mean_to<Dst: Shape>(self) -> Result<Self::With<Dst>, Self::Err>
    where
        Self: MeanInto<Self::With<Dst>, Ax>,
    {
        self.try_mean()
    }
}
impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>, Ax: Axes> MeanTo<Ax> for Tensor<S, E, D, T> {}

impl<S: Shape, D: Device<f32>, T: Tape<D>> Tensor<S, f32, D, T> {
    pub fn mean_along<Ax: Axes>(self) -> Tensor<S::Reduced, f32, D, T>
    where
        S: ReduceShape<Ax>,
    {
        self.try_mean_along().unwrap()
    }

    pub fn try_mean_along<Ax: Axes>(self) -> Result<Tensor<S::Reduced, f32, D, T>, D::Err>
    where
        S: ReduceShape<Ax>,
    {
        self.try_mean()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_valids_mean_axis() {
        let _ = <Tensor1D<5, Cpu> as MeanInto<Tensor0D<Cpu>, _>>::try_mean;
        let _ = <Tensor2D<5, 3, Cpu> as MeanInto<Tensor1D<3, Cpu>, _>>::try_mean;
        let _ = <Tensor2D<5, 3, Cpu> as MeanInto<Tensor1D<5, Cpu>, _>>::try_mean;
        let _ = <Tensor3D<7, 5, 3, Cpu> as MeanInto<Tensor2D<5, 3, Cpu>, _>>::try_mean;
        let _ = <Tensor3D<7, 5, 3, Cpu> as MeanInto<Tensor2D<7, 3, Cpu>, _>>::try_mean;
        let _ = <Tensor3D<7, 5, 3, Cpu> as MeanInto<Tensor2D<7, 5, Cpu>, _>>::try_mean;
        let _ = <Tensor4D<9, 7, 5, 3, Cpu> as MeanInto<Tensor3D<7, 5, 3, Cpu>, _>>::try_mean;
        let _ = <Tensor4D<9, 7, 5, 3, Cpu> as MeanInto<Tensor3D<9, 5, 3, Cpu>, _>>::try_mean;
        let _ = <Tensor4D<9, 7, 5, 3, Cpu> as MeanInto<Tensor3D<9, 7, 3, Cpu>, _>>::try_mean;
        let _ = <Tensor4D<9, 7, 5, 3, Cpu> as MeanInto<Tensor3D<9, 7, 5, Cpu>, _>>::try_mean;
    }

    #[test]
    fn test_mean_1d() {
        let dev = build_test_device!();
        let t = dev.tensor([1.0, 2.0, 3.0]);
        let r = t.trace().mean();
        assert_eq!(r.array(), 2.0);
        // NOTE: .exp() so we cover the case where .mean() has to use result grad.
        let g = r.exp().backward();
        assert_eq!(g.get(&t).array(), [2.463019; 3]);
    }

    #[test]
    fn test_mean_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let r = t.trace().mean();
        assert_eq!(r.array(), 3.5);
        let g = r.backward();
        assert_eq!(g.get(&t).array(), [[1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_mean_3d() {
        let dev = build_test_device!();
        let t = dev.ones::<Rank3<4, 2, 3>>();
        let r = t.trace().mean();
        assert_eq!(r.array(), 1.0);
        let g = r.backward();
        assert_eq!(g.get(&t).array(), [[[1.0 / 24.0; 3]; 2]; 4]);
    }

    #[test]
    fn test_mean_axis_0_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r = t.trace().mean_to::<Rank1<3>>();
        assert_eq!(r.array(), [-0.5, 3.0, -1.5]);
        let g = r.exp().mean().backward();
        assert_eq!(g.get(&t).array(), [[0.10108845, 3.3475895, 0.037188362]; 2]);
    }

    #[test]
    fn test_mean_axis_1_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r = t.trace().mean_to::<Rank1<2>>();
        assert_eq!(r.array(), [2.0, -4.0 / 3.0]);
        let g = r.exp().mean().backward();
        assert_eq!(g.get(&t).array(), [[1.2315094; 3], [0.043932855; 3]]);
    }

    #[test]
    fn test_mean_axes_3d_to_1d_02() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank3<2, 3, 4>>();
        let r = t.trace().mean_to::<Rank1<3>>();
        let r2 = t.trace().sum_to::<Rank2<3, 4>>().sum_to::<Rank1<3>>() / 8.0;
        assert_close(&r.array(), &r2.array());
        let g = r.mean().backward();
        let g2 = r2.mean().backward();
        assert_close(&g.get(&t).array(), &[[[1. / 24.; 4]; 3]; 2]);
        assert_close(&g.get(&t).array(), &g2.get(&t).array());
    }

    #[test]
    fn test_mean_axes_3d_to_1d_01() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank3<2, 3, 4>>();
        let r = t.trace().mean_to::<Rank1<4>>();
        let r2 = t.sum_to::<Rank2<3, 4>>().sum_to::<Rank1<4>>() / 6.0;
        assert_close(&r.array(), &r2.array());
    }
}
