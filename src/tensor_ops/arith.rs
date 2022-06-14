use crate::prelude::*;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Add two [Tensor]s of the same shape together: `lhs + &rhs`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = Tensor2D::ones();
/// let r = add(a, &b); // or `a + &b`
/// assert_eq!(r.data(), &[[2.0, 3.0, 4.0], [0.0, -1.0, -2.0]]);
/// ```
pub fn add<T: Tensor<Dtype = f32>>(lhs: T, rhs: &T::NoTape) -> T {
    fn f(x: &f32, y: &f32) -> f32 {
        x + y
    }
    fn dfdx(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
    fn dfdy(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
    binary_map(lhs, rhs, f, dfdx, dfdy)
}

/// Subtracts two [Tensor]s of the same shape from each other: `lhs - &rhs`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = Tensor2D::ones();
/// let r = sub(a, &b); // or `a - &b`
/// assert_eq!(r.data(), &[[0.0, 1.0, 2.0], [-2.0, -3.0, -4.0]]);
pub fn sub<T: Tensor<Dtype = f32>>(lhs: T, rhs: &T::NoTape) -> T {
    fn f(x: &f32, y: &f32) -> f32 {
        x - y
    }
    fn dfdx(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
    fn dfdy(_x: &f32, _y: &f32) -> f32 {
        -1.0
    }
    binary_map(lhs, rhs, f, dfdx, dfdy)
}

/// Multiplies two [Tensor]s of the same shape together: `lhs * &rhs`.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = Tensor2D::ones();
/// let r = mul(a, &b); // or `a * &b`
/// assert_eq!(r.data(), &[[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
pub fn mul<T: Tensor<Dtype = f32>>(lhs: T, rhs: &T::NoTape) -> T {
    fn f(x: &f32, y: &f32) -> f32 {
        x * y
    }
    fn dfdx(_x: &f32, y: &f32) -> f32 {
        *y
    }
    fn dfdy(x: &f32, _y: &f32) -> f32 {
        *x
    }
    binary_map(lhs, rhs, f, dfdx, dfdy)
}

/// Divides two [Tensor]s of the same shape: `lhs / &rhs`.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = Tensor2D::new([[1.0, 0.5, 1.0], [0.5, 1.0, 3.0]]);
/// let r = div(a, &b); // or `a / &b`
/// assert_eq!(r.data(), &[[1.0, 4.0, 3.0], [-2.0, -2.0, -1.0]]);
pub fn div<T: Tensor<Dtype = f32>>(lhs: T, rhs: &T::NoTape) -> T {
    fn f(x: &f32, y: &f32) -> f32 {
        x * y.recip()
    }
    fn dfdx(_x: &f32, y: &f32) -> f32 {
        y.recip()
    }
    fn dfdy(x: &f32, y: &f32) -> f32 {
        x.neg() * y.powi(2).recip()
    }
    binary_map(lhs, rhs, f, dfdx, dfdy)
}

/// Takes the element wise minimum of two [Tensor]s of the same shape: `min(lhs, &rhs)`.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = Tensor2D::new([[1.0, 0.5, 1.0], [-2.0, 2.0, -3.5]]);
/// let r = minimum(a, &b);
/// assert_eq!(r.data(), &[[1.0, 0.5, 1.0], [-2.0, -2.0, -3.5]]);
pub fn minimum<T: Tensor<Dtype = f32>>(lhs: T, rhs: &T::NoTape) -> T {
    fn f(x: &f32, y: &f32) -> f32 {
        x.min(*y)
    }
    fn dfdx(x: &f32, y: &f32) -> f32 {
        if x < y {
            1.0
        } else if x > y {
            0.0
        } else {
            0.5
        }
    }
    fn dfdy(x: &f32, y: &f32) -> f32 {
        if y < x {
            1.0
        } else if y > x {
            0.0
        } else {
            0.5
        }
    }
    binary_map(lhs, rhs, f, dfdx, dfdy)
}

/// Applies a binary function `f`, it's partial wrt. x `dfdx`, and its partial wrt. y `dfdy`
/// to a pair of [Tensor]s `lhs` and `rhs.
///
/// This is primarily used to implement [add()], [sub()], [mul()], and [div()].
pub fn binary_map<T: Tensor<Dtype = f32>, F, Dfdx, Dfdy>(
    lhs: T,
    rhs: &T::NoTape,
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
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        let result_grad: &T::Array = grads.ref_gradient(&_result);
        T::Device::mul_assign(lhs.mut_data(), result_grad);
        T::Device::mul_assign(rhs_deriv.as_mut(), result_grad);
        T::Device::add_assign(grads.mut_gradient(&lhs), lhs.data());
        T::Device::add_assign(grads.mut_gradient(&_rhs), rhs_deriv.as_ref());
    });
    result.put_tape(tape)
}

macro_rules! binary_ops_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> Add<&$typename<$($Vs, )* NoTape>> for $typename<$($Vs, )* H> {
    type Output = $typename<$($Vs, )* H>;
    /// Calls [add()] - implements `T<H> + &T<NoTape>`
    fn add(self, rhs: &$typename<$($Vs, )* NoTape>) -> Self::Output {
        add(self, rhs)
    }
}

impl<$(const $Vs: usize, )* H: Tape> Sub<&$typename<$($Vs, )* NoTape>> for $typename<$($Vs, )* H> {
    type Output = $typename<$($Vs, )* H>;
    /// Calls [sub()] - implements `T<H> - &T<NoTape>`
    fn sub(self, rhs: &$typename<$($Vs, )* NoTape>) -> Self::Output {
        sub(self, rhs)
    }
}

impl<$(const $Vs: usize, )* H: Tape> Mul<&$typename<$($Vs, )* NoTape>> for $typename<$($Vs, )* H> {
    type Output = $typename<$($Vs, )* H>;
    /// Calls [mul()] - implements `T<H> * &T<NoTape>`
    fn mul(self, rhs: &$typename<$($Vs, )* NoTape>) -> Self::Output {
        mul(self, rhs)
    }
}

impl<$(const $Vs: usize, )* H: Tape> Div<&$typename<$($Vs, )* NoTape>> for $typename<$($Vs, )* H> {
    type Output = $typename<$($Vs, )* H>;
    /// Calls [div()] - implements `T<H> / &T<NoTape>`
    fn div(self, rhs: &$typename<$($Vs, )* NoTape>) -> Self::Output {
        div(self, rhs)
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
    fn test_add_0d() {
        let a = Tensor0D::new(1.0);
        let b = Tensor0D::new(1.0);

        let r = a.trace() + &b;
        assert_eq!(r.data(), &2.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &1.0);
        assert_eq!(gradients.ref_gradient(&b), &1.0);
    }

    #[test]
    fn test_sub_0d() {
        let a = Tensor0D::new(1.0);
        let b = Tensor0D::new(1.0);

        let r = b.trace() - &a;
        assert_eq!(r.data(), &0.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &-1.0);
        assert_eq!(gradients.ref_gradient(&b), &1.0);
    }

    #[test]
    fn test_mul_0d() {
        let a = Tensor0D::new(2.0);
        let b = Tensor0D::new(3.0);

        let r = a.trace() * &b;
        assert_eq!(r.data(), &6.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &3.0);
        assert_eq!(gradients.ref_gradient(&b), &2.0);
    }

    #[test]
    fn test_div_0d() {
        let a = Tensor0D::new(2.0);
        let b = Tensor0D::new(4.0);

        let r = b.trace() / &a;
        assert_eq!(r.data(), &2.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &-1.0);
        assert_eq!(gradients.ref_gradient(&b), &0.5);
    }

    #[test]
    fn test_add_1d() {
        let a = Tensor1D::new([1.0, 2.0, 3.0]);
        let b = Tensor1D::new([1.0, -1.0, 0.0]);

        let r = a.trace() + &b;
        assert_eq!(r.data(), &[2.0, 1.0, 3.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &[1.0 / 3.0; 3]);
    }

    #[test]
    fn test_sub_1d() {
        let a = Tensor1D::new([1.0, 2.0, 3.0]);
        let b = Tensor1D::new([1.0, -1.0, 0.0]);

        let r = b.trace() - &a;
        assert_eq!(r.data(), &[0.0, -3.0, -3.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[-1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &[1.0 / 3.0; 3]);
    }

    #[test]
    fn test_mul_1d() {
        let a = Tensor1D::new([1.0, 2.0, 3.0]);
        let b = Tensor1D::new([1.0, -1.0, 0.0]);

        let r = a.trace() * &b;
        assert_eq!(r.data(), &[1.0, -2.0, 0.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0, -1.0 / 3.0, 0.0]);
        assert_eq!(gradients.ref_gradient(&b), &[1.0 / 3.0, 2.0 / 3.0, 1.0]);
    }

    #[test]
    fn test_div_1d() {
        let a = Tensor1D::new([1.0, 2.0, 3.0]);
        let b = Tensor1D::new([1.0, -1.0, 0.0]);

        let r = b.trace() / &a;
        assert_eq!(r.data(), &[1.0, -0.5, 0.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[-1.0 / 3.0, 1.0 / 12.0, 0.0]);
        assert_eq!(
            gradients.ref_gradient(&b),
            &[1.0 / 3.0, 1.0 / 6.0, 0.11111112]
        );
    }

    #[test]
    fn test_add_2d() {
        let a = Tensor2D::new([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b = Tensor2D::new([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let r = a.trace() + &b;
        assert_eq!(
            r.data(),
            &[[1.1769, 0.5552, 0.5259], [1.3917, 1.0692, 0.873]]
        );
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[1.0 / 6.0; 3]; 2]);
        assert_eq!(gradients.ref_gradient(&b), &[[1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_sub_2d() {
        let a = Tensor2D::new([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b = Tensor2D::new([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let r = b.trace() - &a;
        assert_eq!(
            r.data(),
            &[
                [-0.13709998, 0.21360001, 0.2259],
                [0.2601, -0.33279997, -0.7954]
            ]
        );
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[-1.0 / 6.0; 3]; 2]);
        assert_eq!(gradients.ref_gradient(&b), &[[1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_mul_2d() {
        let a = Tensor2D::new([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b = Tensor2D::new([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let r = a.trace() * &b;
        assert_eq!(
            r.data(),
            &[
                [0.3415743, 0.06565552, 0.056385003],
                [0.46729425, 0.2581082, 0.03236696]
            ]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [0.08665001, 0.06406667, 0.06265],
                [0.13765001, 0.06136667, 0.006466667]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[
                [0.109500006, 0.028466668, 0.025000002],
                [0.0943, 0.11683333, 0.13903335]
            ]
        );
    }

    #[test]
    fn test_div_2d() {
        let a = Tensor2D::new([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b = Tensor2D::new([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let r = b.trace() / &a;
        assert_eq!(
            r.data(),
            &[
                [0.79132426, 2.2505856, 2.506],
                [1.4597031, 0.52524966, 0.046511628]
            ]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [-0.20074181, -2.1961217, -2.7844446],
                [-0.42998204, -0.12488106, -0.009292662]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[
                [0.25367835, 0.97580016, 1.1111112],
                [0.29456818, 0.2377556, 0.1997922]
            ]
        );
    }

    #[test]
    fn test_minimum_0d_rhs() {
        let a = Tensor0D::new(0.0);
        let b = Tensor0D::new(1.0);

        let result = minimum(a.trace(), &b);
        assert_eq!(result.data(), &0.0);

        let gradients = result.backward();
        assert_eq!(gradients.ref_gradient(&a), &1.0);
        assert_eq!(gradients.ref_gradient(&b), &0.0);
    }

    #[test]
    fn test_minimum_0d_lhs() {
        let a = Tensor0D::new(1.0);
        let b = Tensor0D::new(-1.0);

        let result = minimum(a.trace(), &b);
        assert_eq!(result.data(), &-1.0);

        let gradients = result.backward();
        assert_eq!(gradients.ref_gradient(&a), &0.0);
        assert_eq!(gradients.ref_gradient(&b), &1.0);
    }

    #[test]
    fn test_minimum_1d() {
        let a = Tensor1D::new([-1.0, 0.0, 1.0, 2.0]);
        let b = Tensor1D::new([0.0, -1.0, 2.0, 0.0]);

        let result = minimum(a.trace(), &b);
        assert_eq!(result.data(), &[-1.0, -1.0, 1.0, 0.0]);

        // NOTE: call .exp() to make sure we cover cases where it uses the result's gradient
        let gradients = result.exp().sum().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[0.36787945, 0.0, 2.7182817, 0.0]
        );
        assert_eq!(gradients.ref_gradient(&b), &[0.0, 0.36787945, 0.0, 1.0]);
    }

    #[test]
    fn test_minimum_1d_eq() {
        let a = Tensor1D::new([-1.0, 0.0, 1.0]);
        let b = Tensor1D::new([-1.0, 0.0, 1.0]);

        let result = minimum(a.trace(), &b);
        assert_eq!(result.data(), a.data());

        // NOTE: call .exp() to make sure we cover cases where it uses the result's gradient
        let gradients = result.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 6.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &[1.0 / 6.0; 3]);
    }

    #[test]
    fn test_minimum_2d() {
        let a = Tensor2D::new([[-1.1488, 0.2021, 1.1265], [0.1355, -1.7390, -0.4191]]);
        let b = Tensor2D::new([[-0.2851, -0.0965, -0.2529], [0.4759, -0.5239, -0.4651]]);

        let result = minimum(a.trace(), &b);
        assert_eq!(
            result.data(),
            &[[-1.1488, -0.0965, -0.2529], [0.1355, -1.7390, -0.4651]]
        );

        let gradients = result.sum().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
        );
    }
}
