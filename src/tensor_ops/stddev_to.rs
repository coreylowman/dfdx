use crate::{
    arrays::{AxesAsArray, ReduceShape, Shape},
    devices::HasErr,
    gradients::Tape,
    tensor::Tensor,
};

use super::*;

/// Reduces `Axes` of `T` by computing std deviation of all values in those axes.
/// Result [Tensor] has smaller number of dimensions.
///
/// **Pytorch equivalent**: `t.std(Axes, unbiased=False)`
///
/// **Related functions**: [var()], [sqrt()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor2D<2, 3> = tensor([[2.0, 3.0, 4.0], [3.0, 6.0, 9.0]]);
/// let r: Tensor1D<2> = t.stddev(0.0);
/// assert_eq!(r.data(), &[0.6666667_f32.sqrt(), 6.0_f32.sqrt()]);
/// ```
pub trait StddevTo<T, Axes>: HasErr {
    fn stddev(self, epsilon: f32) -> T {
        self.try_stddev(epsilon).unwrap()
    }
    fn try_stddev(self, epsilon: f32) -> Result<T, Self::Err>;
}

impl<Src: Shape, Dst: Shape, Ax, D: Device<f32>, T: Tape<D>> StddevTo<Tensor<Dst, f32, D, T>, Ax>
    for Tensor<Src, f32, D, T>
where
    Self: VarTo<Tensor<Dst, f32, D, T>, Ax, Err = D::Err>,
{
    fn try_stddev(self, epsilon: f32) -> Result<Tensor<Dst, f32, D, T>, Self::Err> {
        self.try_var()?.try_add(epsilon)?.try_sqrt()
    }
}

impl<S: Shape, D: Device<f32>, T: Tape<D>> Tensor<S, f32, D, T> {
    pub fn stddev_along<Ax: AxesAsArray>(self, epsilon: f32) -> Tensor<S::Reduced, f32, D, T>
    where
        S: ReduceShape<Ax>,
    {
        self.try_stddev_along(epsilon).unwrap()
    }

    pub fn try_stddev_along<Ax: AxesAsArray>(
        self,
        epsilon: f32,
    ) -> Result<Tensor<S::Reduced, f32, D, T>, D::Err>
    where
        S: ReduceShape<Ax>,
    {
        self.try_stddev(epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::AsArray;
    use crate::tensor::*;
    use crate::tests::build_test_device;

    #[test]
    fn test_std_axis_0_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r: Tensor1D<4, _, _> = t.trace().stddev(1e-8);
        assert_eq!(r.as_array(), [0.5, 0.0001, 1.0, 3.0]);
        let g = r.mean().backward();
        assert_eq!(
            g.get(&t).as_array(),
            [[0.125, 0.0, -0.125, -0.125], [-0.125, 0.0, 0.125, 0.125]]
        );
    }

    #[test]
    fn test_std_axis_1_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r: Tensor1D<2, _, _> = t.trace().stddev(0.0);
        assert_eq!(r.as_array(), [1.118034, 3.7666297]);
        let g = r.mean().backward();
        assert_eq!(
            g.get(&t).as_array(),
            [
                [-0.16770509, -0.0559017, 0.0559017, 0.16770509],
                [-0.14104122, -0.07466887, 0.024889633, 0.19082046]
            ]
        );
    }
}
