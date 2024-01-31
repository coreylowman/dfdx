use crate::{
    shapes::{Axes, Dtype, ReduceShape, Shape},
    tensor::{Error, Tape, Tensor},
};

use super::{BroadcastTo, Device, MeanTo, TryAdd, TryMul};

/// Normalizes `t` to have stddev `1.0` along `Ax`. `epsilon` is used during stddev.
/// Computes `t / (t.square().mean() + epsilon).sqrt()`.
///
/// Normalizing a single axis:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
/// let _ = t.normalize_rms::<Axis<1>>(1e-5);
/// ```
pub fn normalize_rms<
    Ax: Axes,
    S: Shape + ReduceShape<Ax>,
    E: Dtype,
    D: Device<E>,
    T: Tape<E, D>,
>(
    t: Tensor<S, E, D, T>,
    epsilon: impl Into<f64>,
) -> Tensor<S, E, D, T> {
    t.normalize_rms::<Ax>(epsilon)
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [normalize_rms]
    pub fn normalize_rms<Ax: Axes>(self, epsilon: impl Into<f64>) -> Self
    where
        S: ReduceShape<Ax>,
    {
        self.try_normalize_rms::<Ax>(epsilon).unwrap()
    }

    /// See [normalize_rms]
    pub fn try_normalize_rms<Ax: Axes>(self, epsilon: impl Into<f64>) -> Result<Self, Error>
    where
        S: ReduceShape<Ax>,
    {
        let shape = self.shape;
        let sq = self.retaped::<T>().try_square()?;
        let sq_mean = sq.try_mean::<_, Ax>()?;
        let rsqrt = sq_mean
            .try_add(epsilon)?
            .try_sqrt()?
            .try_recip()?
            .try_broadcast_like(&shape)?;
        self.try_mul(rsqrt)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::*;
    use crate::{shapes::*, tensor::*, tensor_ops::*};

    #[test]
    fn test_1d_normalize_rms_axis_last() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([-2.0, 0.0, 5.0]).to_dtype::<TestDtype>();
        let r = a.leaky_trace().normalize_rms(1e-5);
        assert_close_to_literal!(&r, [-0.64326715, 0.0, 1.6081679]);
        // NOTE: .exp() so we can make sure normalize is using result grad properly
        let g = r.exp().mean().backward();
        assert_close_to_literal!(&g.get(&a), [0.23318729, 0.107211195, 0.09327549]);
    }

    #[test]
    fn test_2d_normalize_rms_axis_last() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[-2.0, 0.0, 5.0], [1.0, 2.0, 3.0]])
            .to_dtype::<TestDtype>();
        let r = a.leaky_trace().normalize_rms::<Axis<1>>(1e-5);
        assert_close_to_literal!(
            r,
            [
                [-0.64326715, 0.0, 1.6081679],
                [0.46290955, 0.9258191, 1.3887286]
            ]
        );
        let g = r.exp().mean().backward();
        assert_close_to_literal!(
            g.get(&a),
            [
                [0.116593644, 0.053605597, 0.046637744],
                [0.019706108, -0.011002079, 0.0007670224]
            ]
        );
    }

    #[test]
    fn test_2d_normalize_rms_axis_first() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[-2.0, 0.0], [1.0, 2.0], [4.0, 5.0]])
            .to_dtype::<TestDtype>();
        let r = a.leaky_trace().normalize_rms::<Axis<0>>(1e-5);
        assert_close_to_literal!(
            r,
            [
                [-0.7559284, 0.0],
                [0.3779642, 0.64326715],
                [1.5118568, 1.6081679]
            ]
        );
        let g = r.exp().mean().backward();
        assert_close_to_literal!(
            g.get(&a),
            [
                [0.14153406, 0.053605597],
                [0.03595103, -0.0043795705],
                [0.061779693, 0.0017521679]
            ]
        );
    }

    #[test]
    fn test_3d_normalize_rms_axis_last() {
        let dev: TestDevice = Default::default();
        let a: Tensor<Rank3<4, 2, 3>, TestDtype, _> = dev.ones();
        let r = a.leaky_trace().normalize_rms::<Axis<2>>(1e-5);
        assert_close_to_literal!(r, [[[1.0; 3]; 2]; 4], 1e-5);
        let g = r.exp().mean().backward();
        assert_close_to_literal!(g.get(&a), [[[0.0; 3]; 2]; 4], 1e-5);
    }
}

// Implementation references:
// - https://github.com/johnma2006/mamba-minimal/blob/03de542a36d873f6e6c4057ad687278cc6ae944d/model.py#L328
// - https://github.com/kroggen/mamba.c/blob/7387f49e352f86a0c22041c0f66fd2a40b58a207/mamba.c#L222
