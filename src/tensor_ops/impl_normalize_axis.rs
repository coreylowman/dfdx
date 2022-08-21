use crate::prelude::*;

/// Normalizes `t` to have mean `0.0` and stddev `1.0` along the `I` dimension of `T`. `epsilon` is passed to [std_axis()].
/// Computes `(t - t.mean(I)) / t.std(I, epsilon)`.
///
/// **Related functions:** [mean_axis()], [std_axis()], [var_axis()]
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// let a = tensor([-2.0, -1.0, 0.0, 5.0, 2.0]);
/// let r = a.normalize_axis::<-1>(1e-5);
/// assert!(r.clone().mean_axis::<-1>().data().abs() < 1e-6);
/// assert!((r.clone().std_axis::<-1>(0.0).data() - 1.0).abs() < 1e-6);
/// ```
pub fn normalize_axis<T, const I: isize>(t: T, epsilon: T::Dtype) -> T
where
    T: Reduce1<I>,
    T::Array: HasAxis<I>,
{
    let (t, tape) = t.split_tape();
    let (std, tape) = std_axis::<T, I>(t.duplicate().put_tape(tape), epsilon)
        .broadcast1()
        .split_tape();
    let (mean, tape) = mean_axis::<T, I>(t.duplicate().put_tape(tape))
        .broadcast1()
        .split_tape();
    let centered = sub(t.put_tape(tape), &mean);
    div(centered, &std)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H>
{
    /// Calls [normalize_axis()] on `self`.
    pub fn normalize_axis<const I: isize>(self, epsilon: f32) -> Self
    where
        Self: Reduce1<I>,
        <Self as HasArrayType>::Array: HasAxis<I>,
    {
        normalize_axis::<Self, I>(self, epsilon)
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0d_normalize_axis_last() {
        let a = tensor(10.0);
        let r = a.trace().normalize_axis::<-1>(1e-5);
        assert_eq!(r.data(), &0.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &0.0);
    }

    #[test]
    fn test_1d_normalize_axis_last() {
        let a = tensor([-2.0, 0.0, 5.0]);
        let r = a.trace().normalize_axis::<-1>(1e-5);
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
        let r = a.trace().normalize_axis::<-1>(1e-5);
        assert_eq!(
            r.data(),
            &[
                [-1.0190487, -0.3396829, 1.3587316],
                [-1.2247356, 0.0, 1.2247356]
            ]
        );
        let gradients = r.exp().mean().backward();
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
        let r = a.trace().normalize_axis::<0>(1e-5);
        assert_eq!(
            r.data(),
            &[
                [-1.2247438, -1.1355485],
                [0.0, -0.16222118],
                [1.2247438, 1.2977698],
            ]
        );
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [0.019245632, 0.025835907],
                [-0.038491584, -0.043060362],
                [0.019245982, 0.017224466],
            ]
        );
    }

    #[test]
    fn test_3d_normalize_axis_last() {
        let a: Tensor3D<4, 2, 3> = TensorCreator::ones();
        let r = a.trace().normalize_axis::<-1>(1e-5);
        assert_eq!(r.data(), &[[[0.0; 3]; 2]; 4]);
        let gradients = r.exp().mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[[0.0; 3]; 2]; 4]);
    }
}
