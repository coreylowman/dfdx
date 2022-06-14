use crate::prelude::*;

/// Reduces the last dimension of the tensor by computing std deviation of all values in the last dimension.
/// Result [Tensor] has smaller number of dimensions.
///
/// Computes: `t.var_last_dim().sqrt()`
///
/// See [var_last_dim()] and [mean_last_dim()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor2D::new([[1.0, 2.0, 3.0], [3.0, 5.0, 8.0]]);
/// let r: Tensor1D<2> = std_last_dim(t);
/// assert_eq!(r.data(), &[1.0, 2.5166113]);
/// ```
pub fn std_last_dim<T: Tensor<Dtype = f32>>(t: T) -> T::LastDimReduced {
    sqrt(var_last_dim(t))
}

/// Reduces the last dimension of the tensor by computing variance of all values in the last dimension.
/// Result [Tensor] has smaller number of dimensions.
///
/// Computes: `(t - t.mean_last_dim()).square().sum_last_dim() / (NUM_ELEMENTS - 1.0)`
///
/// See [std_last_dim()] and [mean_last_dim()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor2D::new([[1.0, 2.0, 3.0], [3.0, 5.0, 8.0]]);
/// let r: Tensor1D<2> = var_last_dim(t);
/// assert_eq!(r.data(), &[1.0, 6.333333]);
/// ```
pub fn var_last_dim<T: Tensor<Dtype = f32>>(t: T) -> T::LastDimReduced {
    let num_elements: f32 = <T::Device as ReduceLastDim<T::Array>>::LAST_DIM as f32;
    let _t: T::NoTape = t.duplicate();
    let (mean, tape) = mean_last_dim(t).split_tape();
    scalar_div(
        sum_last_dim(square(broadcast_inner_sub(_t.put_tape(tape), mean))),
        num_elements - 1.0,
    )
}

macro_rules! std_last_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [std_last_dim()] on `self`.
    pub fn std_last_dim(self) -> <Self as Tensor>::LastDimReduced {
        std_last_dim(self)
    }

    /// Calls [var_last_dim()] on `self`.
    pub fn var_last_dim(self) -> <Self as Tensor>::LastDimReduced {
        var_last_dim(self)
    }
}
    };
}

std_last_impl!(Tensor0D, []);
std_last_impl!(Tensor1D, [M]);
std_last_impl!(Tensor2D, [M, N]);
std_last_impl!(Tensor3D, [M, N, O]);
std_last_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_std_last_0d() {
        let t = Tensor0D::new(3.14);
        let r: Tensor0D<OwnsTape> = t.trace().std_last_dim();
        assert!(r.data().is_nan());
        let gradients = r.mean().backward();
        assert!(gradients.ref_gradient(&t).is_nan());
    }

    #[test]
    fn test_std_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 4.0, 8.0]);
        let r: Tensor0D<OwnsTape> = t.trace().std_last_dim();
        assert_eq!(r.data(), &3.5118847);
        // NOTE: .exp() so we make sure its using result grad properly
        let gradients = r.exp().sum().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[-15.90379, -1.5903792, 17.49417]
        );
    }

    #[test]
    fn test_std_last_2d() {
        let t: Tensor2D<2, 4> = Tensor2D::new([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r: Tensor1D<2, OwnsTape> = t.trace().std_last_dim();
        assert_eq!(r.data(), &[1.2909944, 4.3493295]);
        let gradients = r.sum().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [-0.38729835, -0.12909944, 0.12909944, 0.38729835],
                [-0.3257207, -0.17244038, 0.057480127, 0.44068095]
            ]
        );
    }

    #[test]
    fn test_std_last_3d() {
        let t: Tensor3D<4, 2, 2> = Tensor3D::new([
            [[1.0, 2.0], [5.0, 6.0]],
            [[-2.0, -3.0], [-4.0, -6.0]],
            [[2.0, -1.0], [-6.0, 5.0]],
            [[-2.0, 3.0], [4.0, -5.0]],
        ]);
        let r: Tensor2D<4, 2, OwnsTape> = t.trace().std_last_dim();
        assert_eq!(
            r.data(),
            &[
                [0.70710677, 0.70710677],
                [0.70710677, 1.4142135],
                [2.1213202, 7.7781744],
                [3.535534, 6.363961]
            ]
        );
        let gradients = r.sum().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [[-0.70710677, 0.70710677], [-0.70710677, 0.70710677]],
                [[0.70710677, -0.70710677], [0.70710677, -0.70710677]],
                [[0.7071068, -0.7071068], [-0.7071068, 0.7071068]],
                [[-0.7071068, 0.7071068], [0.7071067, -0.7071067]]
            ]
        );
    }
}
