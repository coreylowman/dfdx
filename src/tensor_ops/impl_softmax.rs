use crate::{
    arrays::{Dtype, HasShape, Shape},
    devices::{device::HasErr, Device},
    gradients::Tape,
    tensor::Tensor,
};

use super::{
    impl_add::TryAdd,
    impl_broadcast_reduce::BroadcastTo,
    impl_max_reduce::MaxTo,
    impl_sub::TrySub,
    impl_sum::SumTo,
    map::{TryExp, TryLn},
};

/// Computes the [LogSumExp](https://en.wikipedia.org/wiki/LogSumExp) function across
/// `Axes`
///
/// **Pytorch equivalent**: `t.exp().sum(Axes).log()`
///
/// **Related functions**: [ln()], [sum()], [exp()], [log_softmax()], [softmax()]
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor3D<2, 4, 6> = TensorCreator::zeros();
/// let _: Tensor2D<2, 4> = t.logsumexp();
/// ```
///
/// Multi axis logsumexp:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor3D<2, 4, 6> = TensorCreator::zeros();
/// let _: Tensor1D<4> = t.logsumexp();
/// ```
pub trait LogSumExpTo<T, Axes>: HasErr {
    fn logsumexp(self) -> T {
        self.try_logsumexp().unwrap()
    }
    fn try_logsumexp(self) -> Result<T, Self::Err>;
}

impl<Src: Shape, Dst: Shape, Axes: Default, E: Dtype, D: Device, T: Tape<D>>
    LogSumExpTo<Tensor<Dst, E, D, T>, Axes> for Tensor<Src, E, D, T>
where
    Tensor<Src, E, D>: MaxTo<Tensor<Dst, E, D>, Axes, Err = D::Err>,
    Tensor<Dst, E, D>: BroadcastTo<Tensor<Src, E, D>, Axes, Err = D::Err>,
    Tensor<Src, E, D, T>: TrySub<Tensor<Src, E, D>, Err = D::Err>
        + TryExp<Err = D::Err>
        + SumTo<Tensor<Dst, E, D, T>, Axes, Err = D::Err>,
    Tensor<Dst, E, D, T>: BroadcastTo<Self, Axes, Err = D::Err>
        + TryAdd<Tensor<Dst, E, D>, Err = D::Err>
        + TryLn<Err = D::Err>,
{
    fn try_logsumexp(self) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        // normalize t
        let max: Tensor<Dst, E, D> = self.with_none_tape().try_max()?;
        let max_b: Tensor<Src, E, D> = max.clone().try_broadcast_to(self.shape())?;
        let t: Self = self.try_sub(max_b)?;

        // do logsumexp
        let t: Self = t.try_exp()?;
        let t: Tensor<Dst, E, D, T> = t.try_sum()?;
        let t: Tensor<Dst, E, D, T> = t.try_ln()?;

        // un-normalize result
        t.try_add(max)
    }
}

pub(crate) mod softmax_internals {
    use super::*;
    use crate::arrays::ReduceShape;

    pub trait LogSoftmaxAlong<Axes>: HasErr {
        fn try_log_softmax_along(self) -> Result<Self, Self::Err>;
    }

    pub trait SoftmaxAlong<Axes>: HasErr {
        fn try_softmax_along(self) -> Result<Self, Self::Err>;
    }

    impl<Src: Shape, Axes, E: Dtype, D: Device, T: Tape<D>> LogSoftmaxAlong<Axes>
        for Tensor<Src, E, D, T>
    where
        Src: ReduceShape<Axes>,
        Self: LogSumExpTo<Tensor<Src::Reduced, E, D, T>, Axes> + TrySub<Self, Err = D::Err>,
        Tensor<Src::Reduced, E, D, T>: BroadcastTo<Self, Axes, Err = D::Err>,
    {
        fn try_log_softmax_along(self) -> Result<Self, Self::Err> {
            let logsumexp: Tensor<Src::Reduced, E, D, T> =
                self.with_empty_tape().try_logsumexp()?;
            let logsumexp: Self = logsumexp.try_broadcast_to(self.shape())?;
            self.try_sub(logsumexp)
        }
    }

    impl<Src: Shape, Axes, E: Dtype, D: Device, T: Tape<D>> SoftmaxAlong<Axes> for Tensor<Src, E, D, T>
    where
        Self: LogSoftmaxAlong<Axes, Err = D::Err> + TryExp<Err = D::Err>,
    {
        fn try_softmax_along(self) -> Result<Self, Self::Err> {
            self.try_log_softmax_along()?.try_exp()
        }
    }
}

/// `log(softmax(t))` in numerically stable way across `Axes`. Does `t - logsumexp(t)` under the hood.
///
/// **Pytorch equivalent**: `t.log_softmax(Axes)`
///
/// **Related functions**: [logsumexp()], [softmax()]
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor3D<2, 3, 5> = TensorCreator::zeros();
/// let _ = t.log_softmax::<Axis<2>>();
/// ```
///
/// Using multi axis log_softmax:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor3D<2, 3, 5> = TensorCreator::zeros();
/// let _ = t.log_softmax::<Axes2<0, 2>>();
/// ```
pub trait TryLogSoftmax: HasErr {
    fn log_softmax<Axes>(self) -> Self
    where
        Self: softmax_internals::LogSoftmaxAlong<Axes>,
    {
        self.try_log_softmax().unwrap()
    }
    fn try_log_softmax<Axes>(self) -> Result<Self, Self::Err>
    where
        Self: softmax_internals::LogSoftmaxAlong<Axes>,
    {
        self.try_log_softmax_along()
    }
}
impl<T: HasErr> TryLogSoftmax for T {}

/// Computes the [softmax function](https://en.wikipedia.org/wiki/Softmax_function) across
/// `Axes`.
///
/// Equivalent to `exp(log_softmax(t))`.
///
/// **Pytorch equivalent**: `t.softmax(Axes)`
///
/// **Related functions**: [logsumexp()], [log_softmax()]
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor3D<2, 3, 5> = TensorCreator::zeros();
/// let _ = t.softmax::<Axis<2>>();
/// ```
///
/// Using multi axis softmax:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor3D<2, 3, 5> = TensorCreator::zeros();
/// let _ = t.softmax::<Axes2<1, 2>>();
/// ```
pub trait TrySoftmax: HasErr {
    fn softmax<Axes>(self) -> Self
    where
        Self: softmax_internals::SoftmaxAlong<Axes>,
    {
        self.try_softmax().unwrap()
    }
    fn try_softmax<Axes>(self) -> Result<Self, Self::Err>
    where
        Self: softmax_internals::SoftmaxAlong<Axes>,
    {
        self.try_softmax_along()
    }
}
impl<T: HasErr> TrySoftmax for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        arrays::{Axes2, Axis},
        devices::{AsArray, Randn},
        tensor::*,
        tensor_ops::{impl_backward::TryBackward, impl_mean::MeanTo},
        tests::{assert_close, build_test_device},
    };

    #[test]
    fn test_logsumexp_1d() {
        let dev = build_test_device!();
        let a = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r: Tensor0D<_, _> = a.trace().logsumexp();
        assert_eq!(r.as_array(), 2.4519143);
        let g = r.backward();
        assert_eq!(
            g.get(&a).as_array(),
            [0.011656231, 0.03168492, 0.08612854, 0.23412165, 0.6364086]
        );
    }

    #[test]
    fn test_logsumexp_2d() {
        let dev = build_test_device!();
        let a: Tensor2D<2, 3, _> = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r: Tensor1D<2, _, _> = a.trace().logsumexp();
        assert_eq!(r.as_array(), [0.40760595, 7.0509458]);
        let g = r.mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
            [
                [0.045015287, 0.12236424, 0.33262047],
                [0.0011778167, 0.023657078, 0.47516513]
            ]
        );
    }

    #[test]
    fn test_log_softmax_1d() {
        let dev = build_test_device!();
        let a = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = a.trace().log_softmax();
        assert_eq!(
            r.as_array(),
            [-4.4519143, -3.4519143, -2.4519143, -1.4519143, -0.4519143]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
            [
                0.18834378,
                0.16831508,
                0.11387146,
                -0.034121647,
                -0.43640864
            ]
        );
    }

    #[test]
    fn test_log_softmax_2d() {
        let dev = build_test_device!();
        let a = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.trace().log_softmax::<Axis<1>>();
        assert_eq!(
            r.as_array(),
            [
                [-2.407606, -1.4076059, -0.40760595],
                [-6.0509458, -3.0509458, -0.05094576]
            ]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
            [
                [0.12165138, 0.044302434, -0.1659538],
                [0.16548885, 0.14300959, -0.30849844]
            ]
        );
    }

    #[test]
    fn test_softmax_1d() {
        let dev = build_test_device!();
        let a = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = a.trace().softmax();
        assert_eq!(
            r.as_array(),
            [0.011656232, 0.031684924, 0.086128555, 0.23412168, 0.6364087]
        );
        let l = r * dev.tensor([0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(l.as_array(), [0.0, 0.0, 0.086128555, 0.0, 0.0]);
        let g = l.mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
            [
                -0.00020078686,
                -0.00054579525,
                0.015742086,
                -0.0040329117,
                -0.010962591
            ]
        );
    }

    #[test]
    fn test_softmax_2d() {
        let dev = build_test_device!();
        let a = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.trace().softmax::<Axis<1>>();
        assert_eq!(
            r.as_array(),
            [
                [0.09003058, 0.24472849, 0.66524094],
                [0.002355633, 0.047314156, 0.9503302]
            ]
        );
        let l = r * dev.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert_eq!(
            l.as_array(),
            [[0.09003058, 0.0, 0.0], [0.0, 0.047314156, 0.0]]
        );
        let g = l.mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
            [
                [0.01365418, -0.0036721744, -0.009982005],
                [-1.85758e-5, 0.0075125876, -0.0074940124]
            ]
        );
    }

    #[test]
    fn test_softmax_2d_0th_axis() {
        let dev = build_test_device!();
        let a = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.trace().softmax::<Axis<0>>();
        assert_eq!(
            r.as_array(),
            [
                [0.047425874, 0.0066928514, 0.0009110514],
                [0.95257413, 0.9933072, 0.9990892]
            ]
        );
        let l = r * dev.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert_eq!(
            l.as_array(),
            [[0.047425874, 0.0, 0.0], [0.0, 0.9933072, 0.0]]
        );
        let g = l.mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
            [
                [0.0075294436, -0.0011080095, 0.0],
                [-0.0075294436, 0.0011080056, 0.0]
            ]
        );
    }

    #[test]
    fn test_softmax_3d_to_1d_12() {
        let dev = build_test_device!();
        let t: Tensor3D<2, 3, 4, _> = dev.randn();
        let r = t.trace().softmax::<Axes2<1, 2>>();
        #[rustfmt::skip]
        assert_close(
            &r.as_array(),
            &[
                [[0.08535644, 0.0987266, 0.00366116, 0.04927256], [0.01169326, 0.1515922, 0.00951258, 0.07721686], [0.0776206, 0.23813945, 0.19471556, 0.00249278]],
                [[0.01881982, 0.25171953, 0.02559674, 0.03725754], [0.04064152, 0.314442, 0.02427996, 0.04708378], [0.02791536, 0.14462142, 0.02221143, 0.04541067]],
            ],
        );
    }
}
