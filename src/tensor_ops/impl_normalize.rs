use crate::prelude::*;

/// `(t - t.mean(-1)) / t.std(-1, epsilon)`. Normalizes `t` to have mean `0.0` and stddev `1.0`.
///
/// `epsilon` is passed to [std_last_dim()].
///
/// See [mean_last_dim()], [std_last_dim()], [var_last_dim()]
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor1D::new([-2.0, -1.0, 0.0, 5.0, 2.0]);
/// let r = normalize(a, 1e-5); // or a.normalize(1e-5);
/// assert!(mean_last_dim(r.duplicate()).data().abs() < 1e-6);
/// assert!((std_last_dim(r.duplicate(), 0.0).data() - 1.0).abs() < 1e-6);
/// ```
pub fn normalize<T: Tensor<Dtype = f32>>(t: T, epsilon: T::Dtype) -> T {
    let (t, tape) = t.split_tape();
    let (std, tape) = std_last_dim(t.duplicate().put_tape(tape), epsilon).split_tape();
    let (mean, tape) = mean_last_dim(t.duplicate().put_tape(tape)).split_tape();
    let centered = sub_broadcast_rhs_last(t.put_tape(tape), &mean);
    div_broadcast_rhs_last(centered, &std)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H>
{
    /// Calls [normalize()] on `self`.
    pub fn normalize(self, epsilon: f32) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0d_normalize() {
        let a = Tensor0D::new(10.0);
        let r = a.trace().normalize(1e-5);
        assert_eq!(r.data(), &0.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &0.0);
    }

    #[test]
    fn test_1d_normalize() {
        let a = Tensor1D::new([-2.0, 0.0, 5.0]);
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
    fn test_2d_normalize() {
        let a: Tensor2D<2, 3> = Tensor2D::new([[-2.0, 0.0, 5.0], [1.0, 2.0, 3.0]]);
        let r = a.trace().normalize(1e-5);
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
    fn test_3d_normalize() {
        let a: Tensor3D<4, 2, 3> = Tensor3D::ones();
        let r = a.trace().normalize(1e-5);
        assert_eq!(r.data(), &[[[0.0; 3]; 2]; 4]);
        let gradients = r.exp().mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[[0.0; 3]; 2]; 4]);
    }
}
