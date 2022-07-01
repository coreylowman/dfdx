use crate::prelude::*;

/// Sums all the values in `self` and divides by number of values.
///
/// Returns a [Tensor0D] (i.e. one number).
pub fn mean<T: Tensor<Dtype = f32>>(t: T) -> Tensor0D<T::Tape> {
    scalar_div(sum(t), T::Array::NUM_ELEMENTS as f32)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [mean()] on `self`.
    pub fn mean(self) -> Tensor0D<<Self as Tensor>::Tape> {
        mean(self)
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
    fn test_mean_0d() {
        let t: Tensor0D = Tensor0D::new(3.0);
        let r = t.trace().mean();
        assert_eq!(r.data(), &3.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&t), &1.0);
    }

    #[test]
    fn test_mean_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let r: Tensor0D<OwnsTape> = t.trace().mean();
        assert_eq!(r.data(), &2.0);
        // NOTE: .exp() so we cover the case where .mean() has to use result grad.
        let gradients = r.exp().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[2.4630187, 2.4630187, 2.4630187]
        );
    }

    #[test]
    fn test_mean_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let r: Tensor0D<OwnsTape> = t.trace().mean();
        assert_eq!(r.data(), &3.5);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&t), &[[1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_mean_3d() {
        let t: Tensor3D<4, 2, 3> = Tensor3D::ones();
        let r: Tensor0D<OwnsTape> = t.trace().mean();
        assert_eq!(r.data(), &1.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&t), &[[[1.0 / 24.0; 3]; 2]; 4]);
    }
}
