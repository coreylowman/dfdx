use super::utils::binary_map;
use crate::prelude::*;

/// Element wise subtraction.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = Tensor2D::ones();
/// let r = sub(a, &b); // or `a - &b`
/// assert_eq!(r.data(), &[[0.0, 1.0, 2.0], [-2.0, -3.0, -4.0]]);
pub fn sub<T: Tensor<Dtype = f32>>(lhs: T, rhs: &T::NoTape) -> T {
    binary_map(lhs, rhs, |x, y| x - y, |_, _| 1.0, |_, _| -1.0)
}

macro_rules! binary_ops_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> std::ops::Sub<&$typename<$($Vs, )* NoneTape>> for $typename<$($Vs, )* H> {
    type Output = $typename<$($Vs, )* H>;
    /// Calls [sub()] - implements `T<H> - &T<NoneTape>`
    fn sub(self, rhs: &$typename<$($Vs, )* NoneTape>) -> Self::Output {
        sub(self, rhs)
    }
}
    };
}

binary_ops_impl!(Tensor0D, []);
binary_ops_impl!(Tensor1D, [N]);
binary_ops_impl!(Tensor2D, [M, N]);
binary_ops_impl!(Tensor3D, [M, N, O]);
binary_ops_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sub_0d() {
        let a = tensor(1.0);
        let b = tensor(1.0);

        let r = b.trace() - &a;
        assert_eq!(r.data(), &0.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &-1.0);
        assert_eq!(gradients.ref_gradient(&b), &1.0);
    }

    #[test]
    fn test_sub_1d() {
        let a = tensor([1.0, 2.0, 3.0]);
        let b = tensor([1.0, -1.0, 0.0]);

        let r = b.trace() - &a;
        assert_eq!(r.data(), &[0.0, -3.0, -3.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[-1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &[1.0 / 3.0; 3]);
    }

    #[test]
    fn test_sub_2d() {
        let a = tensor([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b = tensor([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let r = b.trace() - &a;
        assert_eq!(
            r.data(),
            &[
                [-0.13709998, 0.21360001, 0.2259],
                [0.2601, -0.33279997, -0.7954]
            ]
        );
        let gradients = backward(r.mean());
        assert_eq!(gradients.ref_gradient(&a), &[[-1.0 / 6.0; 3]; 2]);
        assert_eq!(gradients.ref_gradient(&b), &[[1.0 / 6.0; 3]; 2]);
    }
}
