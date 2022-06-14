use crate::prelude::*;
use std::ops::Neg;

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
    fn f(x: &f32, y: &f32) -> f32 {
        x + y
    }
    fn dfdx(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
    fn dfdy(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
    binary_map_broadcast_rhs_last(lhs, rhs, f, dfdx, dfdy)
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
    fn f(x: &f32, y: &f32) -> f32 {
        x - y
    }
    fn dfdx(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
    fn dfdy(_x: &f32, _y: &f32) -> f32 {
        -1.0
    }
    binary_map_broadcast_rhs_last(lhs, rhs, f, dfdx, dfdy)
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
    fn f(x: &f32, y: &f32) -> f32 {
        x * y
    }
    fn dfdx(_x: &f32, y: &f32) -> f32 {
        *y
    }
    fn dfdy(x: &f32, _y: &f32) -> f32 {
        *x
    }
    binary_map_broadcast_rhs_last(lhs, rhs, f, dfdx, dfdy)
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
    fn f(x: &f32, y: &f32) -> f32 {
        x * y.recip()
    }
    fn dfdx(_x: &f32, y: &f32) -> f32 {
        y.recip()
    }
    fn dfdy(x: &f32, y: &f32) -> f32 {
        x.neg() * y.powi(2).recip()
    }
    binary_map_broadcast_rhs_last(lhs, rhs, f, dfdx, dfdy)
}

/// Applies a binary function `f`, it's partial wrt. x `dfdx`, and its partial wrt. y `dfdy`
/// to a pair of [Tensor]s `lhs` and `rhs. Note that `rhs` has it's last dimension reduced,
/// so therefore it's last dimension is broadcasted to `lhs`'s last dimension.
///
/// This is primarily used to implement [add_broadcast_rhs_last()],
/// [sub_broadcast_rhs_last()], [mul_broadcast_rhs_last()], and [div_broadcast_rhs_last()].
pub fn binary_map_broadcast_rhs_last<T: Tensor<Dtype = f32>, F, Dfdx, Dfdy>(
    lhs: T,
    mut rhs: <T::LastDimReduced as Tensor>::NoTape,
    f: F,
    mut dfdx: Dfdx,
    dfdy: Dfdy,
) -> T
where
    F: FnMut(&f32, &f32) -> f32,
    Dfdx: FnMut(&f32, &f32) -> f32,
    Dfdy: FnMut(&f32, &f32) -> f32,
{
    let result = T::NoTape::new_boxed(T::Device::zip_map(lhs.data(), rhs.data(), f));
    let (mut lhs, mut tape) = lhs.split_tape();
    let mut rhs_deriv: Box<T::Array> = T::Device::zip_map(lhs.data(), rhs.data(), dfdy);
    T::Device::zip_map_assign(lhs.mut_data(), rhs.data(), &mut |l, r| *l = dfdx(l, r));
    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        let result_grad: &T::Array = grads.ref_gradient(&_result);
        T::Device::mul_assign(lhs.mut_data(), result_grad);
        T::Device::mul_assign(rhs_deriv.as_mut(), result_grad);
        T::Device::add_assign(grads.mut_gradient(&lhs), lhs.data());
        T::Device::reduce_last_dim_into(rhs_deriv.as_ref(), rhs.mut_data(), &mut |x, y| x + y);
        <T::LastDimReduced as HasDevice>::Device::add_assign(grads.mut_gradient(&rhs), rhs.data());
    });
    result.put_tape(tape)
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
        assert_eq!(gradients.ref_gradient(&b), &[0.62276626, 40.78770447]);
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
                [1.23150945, 3.34758949, 0.02255588],
                [0.0018514994, 15.002855, 0.0002505732]
            ]
        );
        assert_eq!(gradients.ref_gradient(&b), &[-4.60165453, -15.00495720]);
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
                [-0.06131324, -0.02255588, -3.34758949],
                [0.01127794, 1.01520789, 0.0041489224]
            ]
        );
        assert_eq!(gradients.ref_gradient(&b), &[-9.93634319, 10.01206779]);
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
                [-0.06131324, -0.02255588, -3.34758949],
                [0.000111820875, 7342.15527344, 0.0000020480709]
            ]
        );
        assert_eq!(gradients.ref_gradient(&b), &[9.93634319, -73421.55468750]);
    }
}
