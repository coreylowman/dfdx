use crate::{
    arrays::{Axes, HasShape, ReduceShape, Shape},
    gradients::Tape,
    tensor::{HasErr, Tensor},
};

use super::{BroadcastTo, Device, TryDiv, TrySub};

/// Normalizes `t` to have mean `0.0` and stddev `1.0` along `Axes` of `T`. `epsilon` is passed to [stddev()].
/// Computes `(t - t.mean(Axes)) / t.std(Axes, epsilon)`.
///
/// **Related functions:** [mean()], [stddev()], [var()]
///
/// Normalizing a single axis:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor2D<2, 3> = TensorCreator::zeros();
/// let _ = t.normalize::<Axis<1>>(1e-5);
/// ```
pub fn normalize<Ax: Axes, S: Shape + ReduceShape<Ax>, D: Device<f32>, T: Tape<D>>(
    t: Tensor<S, f32, D, T>,
    epsilon: f32,
) -> Tensor<S, f32, D, T> {
    t.normalize_along::<Ax>(epsilon)
}

impl<S: Shape, D: Device<f32>, T: Tape<D>> Tensor<S, f32, D, T> {
    /// See [NormalizeAxes]
    pub fn normalize_along<Ax: Axes>(self, epsilon: f32) -> Self
    where
        S: ReduceShape<Ax>,
    {
        self.try_normalize_along(epsilon).unwrap()
    }

    /// See [NormalizeAxes]
    pub fn try_normalize_along<Ax: Axes>(self, epsilon: f32) -> Result<Self, <Self as HasErr>::Err>
    where
        S: ReduceShape<Ax>,
    {
        let mean = self
            .retaped::<T>()
            .try_mean_along::<Ax>()?
            .try_broadcast_to(self.shape())?;
        let std = self
            .retaped::<T>()
            .try_stddev_along::<Ax>(epsilon)?
            .try_broadcast_to(self.shape())?;
        self.try_sub(mean)?.try_div(std)
    }
}

#[cfg(test)]
mod tests {
    use crate::arrays::Axis;
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_1d_normalize_axis_last() {
        let dev = build_test_device!();
        let a = dev.tensor([-2.0, 0.0, 5.0]);
        let r = a.trace().normalize_along(1e-5);
        assert_eq!(r.array(), [-1.0190487, -0.3396829, 1.3587316]);
        // NOTE: .exp() so we can make sure normalize is using result grad properly
        let g = r.exp().mean().backward();
        assert_eq!(g.get(&a).array(), [0.033410847, -0.04677555, 0.013364702]);
    }

    #[test]
    fn test_2d_normalize_axis_last() {
        let dev = build_test_device!();
        let a = dev.tensor([[-2.0, 0.0, 5.0], [1.0, 2.0, 3.0]]);
        let r = a.trace().normalize_along::<Axis<1>>(1e-5);
        assert_eq!(
            r.array(),
            [
                [-1.0190487, -0.3396829, 1.3587316],
                [-1.2247356, 0.0, 1.2247356]
            ]
        );
        let g = r.exp().mean().backward();
        assert_eq!(
            g.get(&a).array(),
            [
                [0.016705424, -0.023387775, 0.006682351],
                [0.05773133, -0.11547226, 0.057740927]
            ]
        );
    }

    #[test]
    fn test_2d_normalize_axis_first() {
        let dev = build_test_device!();
        let a = dev.tensor([[-2.0, 0.0], [1.0, 2.0], [4.0, 5.0]]);
        let r = a.trace().normalize_along::<Axis<0>>(1e-5);
        assert_eq!(
            r.array(),
            [
                [-1.2247438, -1.1355485],
                [0.0, -0.16222118],
                [1.2247438, 1.2977698],
            ]
        );
        let g = r.exp().mean().backward();
        assert_close(
            &g.get(&a).array(),
            &[
                [0.019245632, 0.025835907],
                [-0.038491584, -0.043060362],
                [0.019245982, 0.01722446],
            ],
        );
    }

    #[test]
    fn test_3d_normalize_axis_last() {
        let dev = build_test_device!();
        let a: Tensor3D<4, 2, 3, _> = dev.ones();
        let r = a.trace().normalize_along::<Axis<2>>(1e-5);
        assert_eq!(r.array(), [[[0.0; 3]; 2]; 4]);
        let g = r.exp().mean().backward();
        assert_eq!(g.get(&a).array(), [[[0.0; 3]; 2]; 4]);
    }
}
