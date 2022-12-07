use super::*;
use crate::{gradients::Tape, shapes::*, tensor::*};

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
///
/// Specifying axes to reduce:
/// ```rust
/// todo!();
/// ```
pub trait LogSumExpTo: HasErr + HasShape {
    fn logsumexp<Dst: Shape, Ax: Axes>(self) -> Self::WithShape<Dst>
    where
        Self::Shape: ReduceShapeTo<Dst, Ax>,
    {
        self.try_logsumexp().unwrap()
    }
    fn try_logsumexp<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ReduceShapeTo<Dst, Ax>;
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> LogSumExpTo for Tensor<S, E, D, T> {
    fn try_logsumexp<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ReduceShapeTo<Dst, Ax>,
    {
        let max: Tensor<Dst, E, D> = self.retaped().try_max()?;
        let t = self.try_sub(max.clone().try_broadcast_like(self.shape())?)?;
        t.try_exp()?.try_sum::<Dst, Ax>()?.try_ln()?.try_add(max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::build_test_device;

    #[test]
    fn test_logsumexp_1d() {
        let dev = build_test_device!();
        let a = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = a.trace().logsumexp();
        assert_eq!(r.array(), 2.4519143);
        let g = r.backward();
        assert_eq!(
            g.get(&a).array(),
            [0.011656231, 0.03168492, 0.08612854, 0.23412165, 0.6364086]
        );
    }

    #[test]
    fn test_logsumexp_2d() {
        let dev = build_test_device!();
        let a = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.trace().logsumexp::<Rank1<2>, _>();
        assert_eq!(r.array(), [0.40760595, 7.0509458]);
        let g = r.mean().backward();
        assert_eq!(
            g.get(&a).array(),
            [
                [0.045015287, 0.12236424, 0.33262047],
                [0.0011778167, 0.023657078, 0.47516513]
            ]
        );
    }
}
