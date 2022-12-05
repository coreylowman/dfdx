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
pub trait LogSumExpInto<T, Axes>: HasErr {
    fn logsumexp(self) -> T {
        self.try_logsumexp().unwrap()
    }
    fn try_logsumexp(self) -> Result<T, Self::Err>;
}

impl<
        Ax: Axes,
        Src: Shape + ReduceShapeTo<Dst, Ax>,
        Dst: Shape,
        E: Dtype,
        D: Device<E>,
        T: Tape<D>,
    > LogSumExpInto<Tensor<Dst, E, D, T>, Ax> for Tensor<Src, E, D, T>
{
    fn try_logsumexp(self) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        // normalize t
        let max: Tensor<Dst, E, D> = self.retaped().try_max()?;
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

pub trait LogSumExpTo<Ax: Axes>: HasShape + HasErr {
    fn logsumexp_to<Dst: Shape>(self) -> Self::WithShape<Dst>
    where
        Self: LogSumExpInto<Self::WithShape<Dst>, Ax>,
    {
        self.logsumexp()
    }

    fn try_logsumexp_to<Dst: Shape>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self: LogSumExpInto<Self::WithShape<Dst>, Ax>,
    {
        self.try_logsumexp()
    }
}
impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>, Ax: Axes> LogSumExpTo<Ax>
    for Tensor<S, E, D, T>
{
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn logsumexp_along<Ax: Axes>(self) -> Tensor<S::Reduced, E, D, T>
    where
        S: ReduceShape<Ax>,
    {
        self.try_logsumexp_along::<Ax>().unwrap()
    }

    pub fn try_logsumexp_along<Ax: Axes>(self) -> Result<Tensor<S::Reduced, E, D, T>, D::Err>
    where
        S: ReduceShape<Ax>,
    {
        self.try_logsumexp()
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
        let r = a.trace().logsumexp_to::<Rank1<2>>();
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
