use crate::arrays::{HasArrayType, HasAxes};
use crate::gradients::Tape;
use crate::prelude::*;

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
pub fn normalize<T, Axes>(t: T, epsilon: T::Dtype) -> T
where
    T: Reduce<Axes>,
    T::Array: HasAxes<Axes>,
{
    let std: T::Reduced = stddev(t.with_empty_tape(), epsilon);
    let mean: T::Reduced = mean(t.with_empty_tape());
    let centered = sub(t, mean.broadcast());
    div(centered, std.broadcast())
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H>
{
    /// Calls [normalize()]
    pub fn normalize<Axes>(self, epsilon: f32) -> Self
    where
        Self: Reduce<Axes>,
        <Self as HasArrayType>::Array: HasAxes<Axes>,
    {
        normalize(self, epsilon)
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
tensor_impl!(Tensor5D, [M, N, O, P, Q]);
tensor_impl!(Tensor6D, [M, N, O, P, Q, R]);

#[cfg(test)]
mod tests {
    use crate::tests::assert_close;

    use super::*;

    #[test]
    fn test_0d_normalize_axis_last() {
        let a = tensor(10.0);
        let r = a.trace().normalize(1e-5);
        assert_eq!(r.data(), &0.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &0.0);
    }

    #[test]
    fn test_1d_normalize_axis_last() {
        let a = tensor([-2.0, 0.0, 5.0]);
        let r = a.trace().normalize(1e-5);
        assert_eq!(r.data(), &[-1.0190487, -0.3396829, 1.3587316]);
        // NOTE: .exp() so we can make sure normalize is using result grad properly
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[0.033410847, -0.04677555, 0.013364702]
        );
    }

    #[test]
    fn test_2d_normalize_axis_last() {
        let a: Tensor2D<2, 3> = tensor([[-2.0, 0.0, 5.0], [1.0, 2.0, 3.0]]);
        let r = a.trace().normalize::<Axis<1>>(1e-5);
        assert_eq!(
            r.data(),
            &[
                [-1.0190487, -0.3396829, 1.3587316],
                [-1.2247356, 0.0, 1.2247356]
            ]
        );
        let gradients = backward(r.exp().mean());
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [0.016705424, -0.023387775, 0.006682351],
                [0.05773133, -0.11547226, 0.057740927]
            ]
        );
    }

    #[test]
    fn test_2d_normalize_axis_first() {
        let a: Tensor2D<3, 2> = tensor([[-2.0, 0.0], [1.0, 2.0], [4.0, 5.0]]);
        let r = a.trace().normalize::<Axis<0>>(1e-5);
        assert_eq!(
            r.data(),
            &[
                [-1.2247438, -1.1355485],
                [0.0, -0.16222118],
                [1.2247438, 1.2977698],
            ]
        );
        let gradients = backward(r.exp().mean());
        assert_close(
            gradients.ref_gradient(&a),
            &[
                [0.019245632, 0.025835907],
                [-0.038491584, -0.043060362],
                [0.019245982, 0.01722446],
            ],
        );
    }

    #[test]
    fn test_3d_normalize_axis_last() {
        let a: Tensor3D<4, 2, 3> = TensorCreator::ones();
        let r = a.trace().normalize::<Axis<2>>(1e-5);
        assert_eq!(r.data(), &[[[0.0; 3]; 2]; 4]);
        let gradients = backward(r.exp().mean());
        assert_eq!(gradients.ref_gradient(&a), &[[[0.0; 3]; 2]; 4]);
    }
}
