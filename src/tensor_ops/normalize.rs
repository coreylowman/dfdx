use crate::{
    gradients::Tape,
    shapes::{Axes, Dtype, HasShape, ReduceShape, Shape},
    tensor::{HasErr, Tensor},
};

use super::{BroadcastTo, Device, MeanTo, StddevTo, TryDiv, TrySub};

/// Normalizes `t` to have mean `0.0` and stddev `1.0` along `Ax`. `epsilon` is used during stddev.
/// Computes `(t - t.mean(Ax)) / t.std(Ax, epsilon)`.
///
/// Normalizing a single axis:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t: Tensor<Rank2<2, 3>> = dev.zeros();
/// let _ = t.normalize::<Axis<1>>(1e-5);
/// ```
pub fn normalize<Ax: Axes, S: Shape + ReduceShape<Ax>, E: Dtype, D: Device<E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
    epsilon: E,
) -> Tensor<S, E, D, T> {
    t.normalize::<Ax>(epsilon)
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    /// See [normalize]
    pub fn normalize<Ax: Axes>(self, epsilon: E) -> Self
    where
        S: ReduceShape<Ax>,
    {
        self.try_normalize(epsilon).unwrap()
    }

    /// See [normalize]
    pub fn try_normalize<Ax: Axes>(self, epsilon: E) -> Result<Self, <Self as HasErr>::Err>
    where
        S: ReduceShape<Ax>,
    {
        let mean = self
            .retaped::<T>()
            .try_mean::<_, Ax>()?
            .try_broadcast_like(self.shape())?;
        let std = self
            .retaped::<T>()
            .try_stddev::<_, Ax>(epsilon)?
            .try_broadcast_like(self.shape())?;
        self.try_sub(mean)?.try_div(std)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::*;
    use crate::{shapes::*, tensor::*, tensor_ops::*};

    #[test]
    fn test_1d_normalize_axis_last() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([-2.0, 0.0, 5.0]);
        let r = a.trace().normalize(1e-5);
        assert_eq!(r.array(), [-1.0190487, -0.3396829, 1.3587316]);
        // NOTE: .exp() so we can make sure normalize is using result grad properly
        let g = r.exp().mean().backward();
        assert_close(&g.get(&a).array(), &[0.033410847, -0.04677555, 0.013364702]);
    }

    #[test]
    fn test_2d_normalize_axis_last() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([[-2.0, 0.0, 5.0], [1.0, 2.0, 3.0]]);
        let r = a.trace().normalize::<Axis<1>>(1e-5);
        assert_eq!(
            r.array(),
            [
                [-1.0190487, -0.3396829, 1.3587316],
                [-1.2247356, 0.0, 1.2247356]
            ]
        );
        let g = r.exp().mean().backward();
        assert_close(
            &g.get(&a).array(),
            &[
                [0.016705424, -0.023387775, 0.006682351],
                [0.05773133, -0.11547226, 0.057740927],
            ],
        );
    }

    #[test]
    fn test_2d_normalize_axis_first() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([[-2.0, 0.0], [1.0, 2.0], [4.0, 5.0]]);
        let r = a.trace().normalize::<Axis<0>>(1e-5);
        assert_close(
            &r.array(),
            &[
                [-1.2247438, -1.1355485],
                [0.0, -0.16222118],
                [1.2247438, 1.2977698],
            ],
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
        let dev: TestDevice = Default::default();
        let a: Tensor<Rank3<4, 2, 3>, TestDtype, _> = dev.ones();
        let r = a.trace().normalize::<Axis<2>>(1e-5);
        assert_eq!(r.array(), [[[0.0; 3]; 2]; 4]);
        let g = r.exp().mean().backward();
        assert_eq!(g.get(&a).array(), [[[0.0; 3]; 2]; 4]);
    }
}
