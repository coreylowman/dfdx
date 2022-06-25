use super::binary_map::{add, binary_map_broadcast_rhs_last, div, mul, sub};
use crate::prelude::*;

/// Add two [Tensor]s together: `lhs + rhs`. `rhs`'s last dimension is broadcasted to be the same size as `lhs`.
///
/// Examples:
/// ```rust
/// use dfdx::prelude::*;
/// let a = Tensor2D::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
/// let b = Tensor1D::new([3.0, 4.0]);
/// let r = add_broadcast_rhs_last(a, b);
/// assert_eq!(r.data(), &[[3.0, 4.0, 5.0], [7.0, 8.0, 9.0]]);
/// ```
pub fn add_broadcast_rhs_last<T: Tensor<Dtype = f32>>(
    lhs: T,
    rhs: <T::LastDimReduced as Tensor>::NoTape,
) -> T {
    binary_map_broadcast_rhs_last(lhs, rhs, add::f, add::dfdx, add::dfdy)
}

/// Subtracts two [Tensor]s: `lhs - rhs`. `rhs`'s last dimension is broadcasted to be the same size as `lhs`.
///
/// Examples:
/// ```rust
/// use dfdx::prelude::*;
/// let a = Tensor2D::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
/// let b = Tensor1D::new([3.0, 4.0]);
/// let r = sub_broadcast_rhs_last(a, b);
/// assert_eq!(r.data(), &[[-3.0, -2.0, -1.0], [-1.0, 0.0, 1.0]]);
/// ```
pub fn sub_broadcast_rhs_last<Lhs: Tensor<Dtype = f32>>(
    lhs: Lhs,
    rhs: <Lhs::LastDimReduced as Tensor>::NoTape,
) -> Lhs {
    binary_map_broadcast_rhs_last(lhs, rhs, sub::f, sub::dfdx, sub::dfdy)
}

/// Multiplies two [Tensor]s: `lhs * rhs`. `rhs`'s last dimension is broadcasted to be the same size as `lhs`.
///
/// Examples:
/// ```rust
/// use dfdx::prelude::*;
/// let a = Tensor2D::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
/// let b = Tensor1D::new([3.0, 4.0]);
/// let r = mul_broadcast_rhs_last(a, b);
/// assert_eq!(r.data(), &[[0.0, 3.0, 6.0], [12.0, 16.0, 20.0]]);
/// ```
pub fn mul_broadcast_rhs_last<T: Tensor<Dtype = f32>>(
    lhs: T,
    rhs: <T::LastDimReduced as Tensor>::NoTape,
) -> T {
    binary_map_broadcast_rhs_last(lhs, rhs, mul::f, mul::dfdx, mul::dfdy)
}

/// Divides two [Tensor]s: `lhs / rhs`. `rhs`'s last dimension is broadcasted to be the same size as `lhs`.
///
/// Examples:
/// ```rust
/// use dfdx::prelude::*;
/// let a = Tensor2D::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
/// let b = Tensor1D::new([3.0, 4.0]);
/// let r = div_broadcast_rhs_last(a, b);
/// assert_eq!(r.data(), &[[0.0, 1.0 / 3.0, 2.0 / 3.0], [3.0 / 4.0, 1.0, 5.0 / 4.0]]);
/// ```
pub fn div_broadcast_rhs_last<T: Tensor<Dtype = f32>>(
    lhs: T,
    rhs: <T::LastDimReduced as Tensor>::NoTape,
) -> T {
    binary_map_broadcast_rhs_last(lhs, rhs, div::f, div::dfdx, div::dfdy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_add_1d() {
        let a: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let b = Tensor0D::new(1.0);
        let r = add_broadcast_rhs_last(a.trace(), b.duplicate());
        assert_eq!(r.data(), &[2.0, 3.0, 4.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &1.0);
    }

    #[test]
    fn test_broadcast_sub_1d() {
        let a: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let b = Tensor0D::new(1.0);
        let r = sub_broadcast_rhs_last(a.trace(), b.duplicate());
        assert_eq!(r.data(), &[0.0, 1.0, 2.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &-1.0);
    }

    #[test]
    fn test_broadcast_mul_1d() {
        let a: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let b = Tensor0D::new(1.0);
        let r = mul_broadcast_rhs_last(a.trace(), b.duplicate());
        assert_eq!(r.data(), &[1.0, 2.0, 3.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &2.0);
    }

    #[test]
    fn test_broadcast_div_1d() {
        let a: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let b = Tensor0D::new(1.0);
        let r = div_broadcast_rhs_last(a.trace(), b.duplicate());
        assert_eq!(r.data(), &[1.0, 2.0, 3.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &-2.0);
    }

    #[test]
    fn test_broadcast_add_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, -3.0], [-4.0, 5.0, -6.0]]);
        let b: Tensor1D<2> = Tensor1D::new([-1.0, 0.5]);
        let r = add_broadcast_rhs_last(a.trace(), b.duplicate());
        assert_eq!(r.data(), &[[0.0, 1.0, -4.0], [-3.5, 5.5, -5.5]]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [0.16666667, 0.45304698, 0.0030526067],
                [0.0050328975, 40.78199, 0.0006811286]
            ]
        );
        assert_eq!(gradients.ref_gradient(&b), &[0.62276626, 40.787705]);
    }

    #[test]
    fn test_broadcast_sub_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, -3.0], [-4.0, 5.0, -6.0]]);
        let b: Tensor1D<2> = Tensor1D::new([-1.0, 0.5]);
        let r = sub_broadcast_rhs_last(a.trace(), b.duplicate());
        assert_eq!(r.data(), &[[2.0, 3.0, -2.0], [-4.5, 4.5, -6.5]]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [1.2315095, 3.3475895, 0.02255588],
                [0.0018514994, 15.002855, 0.0002505732]
            ]
        );
        assert_eq!(gradients.ref_gradient(&b), &[-4.6016545, -15.004957]);
    }

    #[test]
    fn test_broadcast_mul_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, -3.0], [-4.0, 5.0, -6.0]]);
        let b: Tensor1D<2> = Tensor1D::new([-1.0, 0.5]);
        let r = mul_broadcast_rhs_last(a.trace(), b.duplicate());
        assert_eq!(r.data(), &[[-1.0, -2.0, 3.0], [-2.0, 2.5, -3.0]]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [-0.06131324, -0.02255588, -3.3475895],
                [0.01127794, 1.0152079, 0.0041489224]
            ]
        );
        assert_eq!(gradients.ref_gradient(&b), &[-9.936343, 10.012068]);
    }

    #[test]
    fn test_broadcast_div_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, -3.0], [-4.0, 5.0, -6.0]]);
        let b: Tensor1D<2> = Tensor1D::new([-1.0, 0.5]);
        let r = div_broadcast_rhs_last(a.trace(), b.duplicate());
        assert_eq!(r.data(), &[[-1.0, -2.0, 3.0], [-8.0, 10.0, -12.0]]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [-0.06131324, -0.02255588, -3.3475895],
                [0.000111820875, 7342.1553, 0.0000020480709]
            ]
        );
        assert_eq!(gradients.ref_gradient(&b), &[9.936343, -73421.555]);
    }
}
