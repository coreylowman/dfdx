use crate::prelude::*;

/// `t.mean(-1)`. Reduces the last dimension of the tensor by taking mean of all values in the last dimension.
/// Result [Tensor] has smaller number of dimensions.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let r: Tensor1D<2> = mean_last_dim(t);
/// assert_eq!(r.data(), &[2.0, 5.0]);
/// ```
pub fn mean_last_dim<T: Tensor<Dtype = f32>>(t: T) -> T::LastDimReduced {
    div_scalar(
        sum_last_dim(t),
        <T::Device as ReduceLastDim<T::Array>>::LAST_DIM as f32,
    )
}

macro_rules! mean_last_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [mean_last_dim()] on `self`.
    pub fn mean_last_dim(self) -> <Self as Tensor>::LastDimReduced {
        mean_last_dim(self)
    }
}
    };
}

mean_last_impl!(Tensor0D, []);
mean_last_impl!(Tensor1D, [M]);
mean_last_impl!(Tensor2D, [M, N]);
mean_last_impl!(Tensor3D, [M, N, O]);
mean_last_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_last_0d() {
        let t = Tensor0D::new(2.0);
        let r: Tensor0D<OwnedTape> = t.trace().mean_last_dim();
        assert_eq!(r.data(), &2.0);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &1.0);
    }

    #[test]
    fn test_mean_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let r: Tensor0D<OwnedTape> = t.trace().mean_last_dim();
        assert_eq!(r.data(), &2.0);
        // NOTE: .exp() so we make sure its using result grad properly
        let gradients = r.exp().mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[2.4630187; 3]);
    }

    #[test]
    fn test_mean_last_2d() {
        let t: Tensor2D<2, 4> = Tensor2D::new([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0]]);
        let r: Tensor1D<2, OwnedTape> = t.trace().mean_last_dim();
        assert_eq!(r.data(), &[2.5, 5.5]);
        let gradients = r.sum().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
        );
    }

    #[test]
    fn test_mean_last_3d() {
        let t: Tensor3D<4, 2, 3> = Tensor3D::new([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            [[-3.0, 2.0, -1.0], [-6.0, 5.0, -4.0]],
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
        ]);
        let r: Tensor2D<4, 2, OwnedTape> = t.trace().mean_last_dim();
        assert_eq!(
            r.data(),
            &[
                [2.0, 5.0],
                [-2.0, -5.0],
                [-2.0 / 3.0, -5.0 / 3.0],
                [2.0 / 3.0, 5.0 / 3.0]
            ]
        );
        let gradients = r.sum().backward();
        assert_eq!(gradients.ref_gradient(&t), &[[[1.0 / 3.0; 3]; 2]; 4]);
    }
}
