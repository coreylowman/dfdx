use crate::prelude::*;

/// Clamp all elements between the provided min and max values.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-1.0, -0.5, 0.0, 0.5, 1.0]);
/// let r = t.clamp(-0.5, 0.5);
/// assert_eq!(r.data(), &[-0.5, -0.5, 0.0, 0.5, 0.5]);
/// ```
pub fn clamp<T: Tensor<Dtype = f32>>(t: T, min: T::Dtype, max: T::Dtype) -> T {
    map(
        t,
        move |x| x.clamp(min, max),
        move |x| if (min..=max).contains(x) { 1.0 } else { 0.0 },
    )
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [clamp()] on self
    pub fn clamp(self, min: f32, max: f32) -> Self {
        clamp(self, min, max)
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
    fn test_clamp_0d() {
        let t = Tensor0D::new(1.0);
        let r = t.trace().clamp(0.0, 1.0);
        assert_eq!(r.data(), &1.0);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &1.0);
    }

    #[test]
    fn test_clamp_1d() {
        let t = Tensor1D::new([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]);
        let r = t.trace().clamp(-0.5, 0.25);
        assert_eq!(r.data(), &[-0.5, -0.5, -0.25, 0.0, 0.25, 0.25, 0.25]);
        // NOTE: .exp() so we cover case where .clamp() needs to use result's grad
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[0.0, 0.08664724, 0.11125726, 0.14285715, 0.1834322, 0.0, 0.0]
        );
    }

    #[test]
    fn test_clamp_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[-1.0, 0.0, 1.0], [-2.0, 2.0, 1.1]]);
        let r = t.trace().clamp(-1.0, 1.0);
        assert_eq!(r.data(), &[[-1.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[[1.0 / 6.0; 3], [0.0; 3]]);
    }
}
