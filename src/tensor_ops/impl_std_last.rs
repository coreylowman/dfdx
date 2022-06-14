use crate::prelude::*;

/// Reduces the last dimension of the tensor by computing std deviation of all values in the last dimension.
/// Result [Tensor] has smaller number of dimensions.
///
/// Computes: `t.var_last_dim().sqrt()`
///
/// See [var_last_dim()] and [sqrt()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor2D::new([[2.0, 3.0, 4.0], [3.0, 6.0, 9.0]]);
/// let r: Tensor1D<2> = std_last_dim(t);
/// assert_eq!(r.data(), &[0.6666667_f32.sqrt(), 6.0_f32.sqrt()]);
/// ```
pub fn std_last_dim<T: Tensor<Dtype = f32>>(t: T) -> T::LastDimReduced {
    sqrt(var_last_dim(t))
}

/// Reduces the last dimension of the tensor by computing variance of all values in the last dimension.
/// Result [Tensor] has smaller number of dimensions.
///
/// Computes: `(t - t.mean_last_dim()).square().sum_last_dim() / NUM_ELEMENTS`
///
/// See [std_last_dim()] and [mean_last_dim()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor2D::new([[2.0, 3.0, 4.0], [3.0, 6.0, 9.0]]);
/// let r: Tensor1D<2> = var_last_dim(t);
/// assert_eq!(r.data(), &[0.6666667, 6.0]);
/// ```
///
/// Note: equivalent to pytorch: `t.var(-1, unbiased=False)`.
pub fn var_last_dim<T: Tensor<Dtype = f32>>(t: T) -> T::LastDimReduced {
    let num_elements: f32 = <T::Device as ReduceLastDim<T::Array>>::LAST_DIM as f32;
    let _t: T::NoTape = t.duplicate();
    let (mean, tape) = mean_last_dim(t).split_tape();
    scalar_div(
        sum_last_dim(square(sub_broadcast_rhs_last(_t.put_tape(tape), mean))),
        num_elements,
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
    fn test_var_last_0d() {
        let t = Tensor0D::new(3.14);
        let r: Tensor0D<OwnsTape> = t.trace().var_last_dim();
        assert_eq!(r.data(), &0.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&t), &0.0);
    }

    #[test]
    fn test_std_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 4.0, 8.0]);
        let r: Tensor0D<OwnsTape> = t.trace().std_last_dim();
        assert_eq!(r.data(), &2.867442);
        // NOTE: .exp() so we make sure its using result grad properly
        let gradients = r.exp().sum().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[-6.8167453, -0.6816746, 7.4984202]
        );
    }

    #[test]
    fn test_std_last_2d() {
        let t: Tensor2D<2, 4> = Tensor2D::new([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r: Tensor1D<2, OwnsTape> = t.trace().std_last_dim();
        assert_eq!(r.data(), &[1.118034, 3.7666297]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [-0.16770509, -0.0559017, 0.0559017, 0.16770509],
                [-0.14104122, -0.07466887, 0.024889633, 0.19082046]
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
        assert_eq!(r.data(), &[[0.5, 0.5], [0.5, 1.0], [1.5, 5.5], [2.5, 4.5]]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [[-0.0625, 0.0625], [-0.0625, 0.0625]],
                [[0.0625, -0.0625], [0.0625, -0.0625]],
                [[0.0625, -0.0625], [-0.0625, 0.0625]],
                [[-0.0625, 0.0625], [0.0625, -0.0625]]
            ]
        );
    }
}
