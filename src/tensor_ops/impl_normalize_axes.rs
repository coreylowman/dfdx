use crate::prelude::*;

/// Normalizes `t` to have mean `0.0` and stddev `1.0` along `Axes` of `T`. `epsilon` is passed to [std_axes()].
/// Computes `(t - t.mean(Axes)) / t.std(Axes, epsilon)`.
///
/// **Related functions:** [mean_axes()], [std_axes()], [var_axes()]
///
/// Normalizing a single axis:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = tensor([-2.0, -1.0, 0.0, 5.0, 2.0]);
/// let r = a.normalize_axis::<-1>(1e-5);
/// assert!(r.clone().mean_axis::<-1>().data().abs() < 1e-6);
/// assert!((r.clone().std_axis::<-1>(0.0).data() - 1.0).abs() < 1e-6);
/// ```
pub fn normalize_axes<T, Axes>(t: T, epsilon: T::Dtype) -> T
where
    T: Reduce<Axes>,
    T::Array: HasAxes<Axes>,
{
    let (t, tape) = t.split_tape();
    let (std, tape) = std_axes(t.duplicate().put_tape(tape), epsilon)
        .broadcast()
        .split_tape();
    let (mean, tape) = mean_axes(t.duplicate().put_tape(tape))
        .broadcast()
        .split_tape();
    let centered = sub(t.put_tape(tape), &mean);
    div(centered, &std)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H>
{
    /// Calls [normalize_axes()] on `self` with `Axis<I>`
    pub fn normalize_axis<const I: isize>(self, epsilon: f32) -> Self
    where
        Self: Reduce<Axis<I>>,
        <Self as HasArrayType>::Array: HasAxes<Axis<I>>,
    {
        normalize_axes(self, epsilon)
    }
    /// Calls [normalize_axes()] on `self`.
    pub fn normalize_axes<Axes>(self, epsilon: f32) -> Self
    where
        Self: Reduce<Axes>,
        <Self as HasArrayType>::Array: HasAxes<Axes>,
    {
        normalize_axes(self, epsilon)
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
    use crate::tests::assert_close;

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

    #[test]
    fn test_normalize_axes_3d_to_1d() {
        let t = tensor([
            [
                [0.00590846, 2.0021265, -1.8343164, 1.8991678],
                [1.5907278, 2.7033613, -2.3023937, 0.04476346],
                [-1.994531, -0.42200476, 1.946085, -0.07927357],
            ],
            [
                [0.876994, -0.15917207, 1.5000577, 2.4590983],
                [0.13156284, -1.6969906, 0.98267025, -0.68137807],
                [-2.1179097, 0.29831272, 0.63086104, 0.927469],
            ],
        ]);
        let r = t.trace().normalize_axes::<Axes2<0, 2>>(1e-5);
        assert_close(
            r.data(),
            &[
                [
                    [-0.6249793, 0.8641092, -1.9977042, 0.78730667],
                    [0.9596546, 1.6742531, -1.5407361, -0.03325422],
                    [-1.4466574, -0.24501033, 1.5645673, 0.01688794],
                ],
                [
                    [0.02481116, -0.7481219, 0.48958856, 1.2049895],
                    [0.02249343, -1.1519107, 0.5691245, -0.49962488],
                    [-1.5409371, 0.3054207, 0.5595378, 0.7861912],
                ],
            ],
        );
        let g = r.exp().mean().backward();
        assert_close(
            g.ref_gradient(&t),
            &[
                [
                    [-0.01023664, 0.00252886, 0.01822704, -0.00063605],
                    [-0.00979434, 0.03576495, 0.02172576, -0.0157913],
                    [0.01457185, -0.01464859, 0.04239608, -0.01735426],
                ],
                [
                    [-0.0143645, -0.00849663, -0.00935561, 0.02233355],
                    [-0.01643993, 0.00957545, -0.01744244, -0.00759815],
                    [0.01755754, -0.01772431, -0.01508666, -0.00971166],
                ],
            ],
        );
    }
}
