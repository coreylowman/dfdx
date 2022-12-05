use super::{BroadcastTo, Device, TrySub};
use crate::{gradients::Tape, shapes::*, tensor::Tensor};

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
pub fn log_softmax<Ax: Axes, S: Shape, E: Dtype, D: Device<E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T>
where
    S: ReduceShape<Ax>,
{
    t.log_softmax::<Ax>()
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn log_softmax<Ax: Axes>(self) -> Self
    where
        S: ReduceShape<Ax>,
    {
        self.try_log_softmax::<Ax>().unwrap()
    }

    pub fn try_log_softmax<Ax: Axes>(self) -> Result<Self, D::Err>
    where
        S: ReduceShape<Ax>,
    {
        let logsumexp = self.retaped::<T>().try_logsumexp_along::<Ax>()?;
        let logsumexp: Self = logsumexp.try_broadcast_to(self.shape())?;
        self.try_sub(logsumexp)
    }
}

#[cfg(test)]
mod tests {
    use crate::{shapes::Axis, tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_log_softmax_1d() {
        let dev = build_test_device!();
        let a = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = a.trace().log_softmax();
        assert_eq!(
            r.array(),
            [-4.4519143, -3.4519143, -2.4519143, -1.4519143, -0.4519143]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.get(&a).array(),
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
            r.array(),
            [
                [-2.407606, -1.4076059, -0.40760595],
                [-6.0509458, -3.0509458, -0.05094576]
            ]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.get(&a).array(),
            [
                [0.12165138, 0.044302434, -0.1659538],
                [0.16548885, 0.14300959, -0.30849844]
            ]
        );
    }
}
