use super::utils::map;
use crate::gradients::Tape;
use crate::prelude::*;

/// Raises to a float power. `t^i`.
///
/// The derivative is `i * t.powf(i - 1)`.
///
/// **Related functions**: [powi()], [sqrt()]
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.powf(-3.2);
/// ```
pub fn powf<T: Tensor<Dtype = f32>>(t: T, i: T::Dtype) -> T {
    map(t, move |x| x.powf(i), move |x| i * x.powf(i - 1.0))
}

/// Raises to an integer power. `t^i`.
///
/// The derivative is `i * t.powi(i - 1)`.
///
/// **Related functions**: [powf()], [square()]
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.powi(3);
/// ```
pub fn powi<T: Tensor<Dtype = f32>>(t: T, i: i32) -> T {
    map(t, move |x| x.powi(i), move |x| i as f32 * x.powi(i - 1))
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [powf()] on `self`.
    pub fn powf(self, i: f32) -> Self {
        powf(self, i)
    }

    /// Calls [powi()] on `self`.
    pub fn powi(self, i: i32) -> Self {
        powi(self, i)
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
    fn test_powf_positive() {
        let t = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.trace().powf(3.5);
        assert!(r.data()[0].is_nan());
        assert!(r.data()[1].is_nan());
        assert_eq!(&r.data()[2..], &[0.0, 1.0, 11.313708]);

        let g = backward(r.sum());
        let grad = g.ref_gradient(&t);
        assert!(grad[0].is_nan());
        assert!(grad[1].is_nan());
        assert_eq!(&grad[2..], &[0.0, 3.5, 19.79899]);
    }

    #[test]
    fn test_powf_negative() {
        let t = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.trace().powf(-1.2);
        assert!(r.data()[0].is_nan());
        assert!(r.data()[1].is_nan());
        assert_eq!(&r.data()[2..], &[f32::INFINITY, 1.0, 0.43527526]);

        let g = backward(r.sum());
        let grad = g.ref_gradient(&t);
        assert!(grad[0].is_nan());
        assert!(grad[1].is_nan());
        assert_eq!(&grad[2..], &[f32::NEG_INFINITY, -1.2, -0.26116517]);
    }

    #[test]
    fn test_powi_positive() {
        let t = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.trace().powi(3);
        assert_eq!(r.data(), &[-8., -1., 0., 1., 8.]);
        let g = backward(r.sum());
        assert_eq!(g.ref_gradient(&t), &[12., 3., 0., 3., 12.]);
    }

    #[test]
    fn test_powi_negative() {
        let t = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = t.trace().powi(-3);
        assert_eq!(r.data(), &[-0.125, -1.0, f32::INFINITY, 1.0, 0.125]);
        let g = backward(r.sum());
        assert_eq!(
            g.ref_gradient(&t),
            &[-0.1875, -3., f32::NEG_INFINITY, -3., -0.1875]
        );
    }
}
