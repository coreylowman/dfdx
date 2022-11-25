use crate::{
    arrays::{Dtype, HasShape, ReduceShape, Shape},
    devices::{Device, HasErr},
    gradients::Tape,
    tensor::Tensor,
    tensor_ops::{BroadcastTo, StddevTo, TryDiv, TryMeanTo, TrySub},
};

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
pub trait NormalizeAxes<Axes>: HasErr {
    fn try_normalize_axes(self, epsilon: f32) -> Result<Self, Self::Err>;
}

impl<Axes, Src: Shape + ReduceShape<Axes>, E: Dtype, D: Device, T: Tape<D>> NormalizeAxes<Axes>
    for Tensor<Src, E, D, T>
where
    Self: TryMeanTo<Tensor<Src::Reduced, E, D, T>, Axes>
        + StddevTo<Tensor<Src::Reduced, E, D, T>, Axes>
        + TrySub
        + TryDiv,
    Tensor<Src::Reduced, E, D, T>: BroadcastTo<Self, Axes, Err = Self::Err>,
{
    fn try_normalize_axes(self, epsilon: f32) -> Result<Self, Self::Err> {
        let mean = self
            .with_empty_tape()
            .try_mean()?
            .try_broadcast_to(self.shape())?;
        let std = self
            .with_empty_tape()
            .try_stddev(epsilon)?
            .try_broadcast_to(self.shape())?;
        self.try_sub(mean)?.try_div(std)
    }
}

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> Tensor<S, E, D, T> {
    /// See [NormalizeAxes]
    pub fn normalize<Axes>(self, epsilon: f32) -> Self
    where
        Self: NormalizeAxes<Axes>,
    {
        self.try_normalize(epsilon).unwrap()
    }

    /// See [NormalizeAxes]
    pub fn try_normalize<Axes>(self, epsilon: f32) -> Result<Self, <Self as HasErr>::Err>
    where
        Self: NormalizeAxes<Axes>,
    {
        self.try_normalize_axes(epsilon)
    }
}

#[cfg(test)]
mod tests {
    use crate::arrays::Axis;
    use crate::devices::{AsArray, Ones};
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_1d_normalize_axis_last() {
        let dev = build_test_device!();
        let a = dev.tensor([-2.0, 0.0, 5.0]);
        let r = a.trace().normalize(1e-5);
        assert_eq!(r.as_array(), [-1.0190487, -0.3396829, 1.3587316]);
        // NOTE: .exp() so we can make sure normalize is using result grad properly
        let g = r.exp().mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
            [0.033410847, -0.04677555, 0.013364702]
        );
    }

    #[test]
    fn test_2d_normalize_axis_last() {
        let dev = build_test_device!();
        let a = dev.tensor([[-2.0, 0.0, 5.0], [1.0, 2.0, 3.0]]);
        let r = a.trace().normalize::<Axis<1>>(1e-5);
        assert_eq!(
            r.as_array(),
            [
                [-1.0190487, -0.3396829, 1.3587316],
                [-1.2247356, 0.0, 1.2247356]
            ]
        );
        let g = r.exp().mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
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
        let r = a.trace().normalize::<Axis<0>>(1e-5);
        assert_eq!(
            r.as_array(),
            [
                [-1.2247438, -1.1355485],
                [0.0, -0.16222118],
                [1.2247438, 1.2977698],
            ]
        );
        let g = r.exp().mean().backward();
        assert_close(
            &g.get(&a).as_array(),
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
        let r = a.trace().normalize::<Axis<2>>(1e-5);
        assert_eq!(r.as_array(), [[[0.0; 3]; 2]; 4]);
        let g = r.exp().mean().backward();
        assert_eq!(g.get(&a).as_array(), [[[0.0; 3]; 2]; 4]);
    }
}
