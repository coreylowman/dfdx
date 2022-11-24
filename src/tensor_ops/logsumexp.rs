use crate::{
    arrays::{Dtype, HasShape, Shape},
    devices::{device::HasErr, Device},
    gradients::Tape,
    tensor::Tensor,
};

use super::*;

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
        + Exp<Err = D::Err>
        + SumTo<Tensor<Dst, E, D, T>, Axes, Err = D::Err>,
    Tensor<Dst, E, D, T>: BroadcastTo<Self, Axes, Err = D::Err>
        + TryAdd<Tensor<Dst, E, D>, Err = D::Err>
        + Ln<Err = D::Err>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{devices::AsArray, tensor::*, tests::build_test_device};

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
}
