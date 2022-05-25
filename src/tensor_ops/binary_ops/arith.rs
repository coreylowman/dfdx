use crate::prelude::*;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Add two [Tensor]s of the same shape together: `&lhs + rhs`
pub fn add<T: Tensor<Dtype = f32>>(lhs: &T::NoTape, rhs: T) -> T {
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

/// Subtracts two [Tensor]s of the same shape from each other: `&lhs - rhs`
pub fn sub<T: Tensor<Dtype = f32>>(lhs: &T::NoTape, rhs: T) -> T {
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

/// Multiplies two [Tensor]s of the same shape together: `&lhs * rhs`.
pub fn mul<T: Tensor<Dtype = f32>>(lhs: &T::NoTape, rhs: T) -> T {
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

/// Divides two [Tensor]s of the same shape: `&lhs / rhs`.
pub fn div<T: Tensor<Dtype = f32>>(lhs: &T::NoTape, rhs: T) -> T {
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

/// Applies a binary function `f`, it's partial wrt. x `dfdx`, and its partial wrt. y `dfdy`
/// to a pair of [Tensor]s `lhs` and `rhs.
///
/// This is primarily used to implement [add()], [sub()], [mul()], and [div()].
pub fn binary_map<T: Tensor<Dtype = f32>, F, Dfdx, Dfdy>(
    lhs: &T::NoTape,
    rhs: T,
    f: F,
    dfdx: Dfdx,
    dfdy: Dfdy,
) -> T
where
    F: FnMut(&f32, &f32) -> f32 + Copy,
    Dfdx: FnMut(&f32, &f32) -> f32 + Copy,
    Dfdy: FnMut(&f32, &f32) -> f32 + Copy,
{
    let result = T::NoTape::new_boxed(T::Device::zip_map(lhs.data(), rhs.data(), f));
    let mut lhs_deriv = T::Device::zip_map(lhs.data(), rhs.data(), dfdx);
    let mut rhs_deriv = T::Device::zip_map(lhs.data(), rhs.data(), dfdy);
    let (rhs, mut tape_holder) = rhs.split_tape_holder();
    let _lhs = lhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let result_grad = tape.ref_gradient(&_result);
        T::Device::mul_assign(lhs_deriv.as_mut(), result_grad);
        T::Device::mul_assign(rhs_deriv.as_mut(), result_grad);
        T::Device::add_assign(tape.mut_gradient(&_lhs), lhs_deriv.as_ref());
        T::Device::add_assign(tape.mut_gradient(&rhs), rhs_deriv.as_ref());
    });
    result.with_tape_holder(tape_holder)
}

macro_rules! binary_ops_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> Add<$typename<$($Vs, )* H>> for &$typename<$($Vs, )* NoTape> {
    type Output = $typename<$($Vs, )* H>;
    /// Calls [add()] - implements `&T<NoTape> + T<H>`
    fn add(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
        add(self, rhs)
    }
}

impl<$(const $Vs: usize, )* H: TapeHolder> Sub<$typename<$($Vs, )* H>> for &$typename<$($Vs, )* NoTape> {
    type Output = $typename<$($Vs, )* H>;
    /// Calls [sub()] - implements `&T<NoTape> - T<H>`
    fn sub(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
        sub(self, rhs)
    }
}

impl<$(const $Vs: usize, )* H: TapeHolder> Mul<$typename<$($Vs, )* H>> for &$typename<$($Vs, )* NoTape> {
    type Output = $typename<$($Vs, )* H>;
    /// Calls [mul()] - implements `&T<NoTape> * T<H>`
    fn mul(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
        mul(self, rhs)
    }
}

impl<$(const $Vs: usize, )* H: TapeHolder> Div<$typename<$($Vs, )* H>> for &$typename<$($Vs, )* NoTape> {
    type Output = $typename<$($Vs, )* H>;
    /// Calls [div()] - implements `&T<NoTape> / T<H>`
    fn div(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
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

        let r = &b + a.trace();
        assert_eq!(r.data(), &2.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &1.0);
        assert_eq!(gradients.ref_gradient(&b), &1.0);
    }

    #[test]
    fn test_sub_0d() {
        let a = Tensor0D::new(1.0);
        let b = Tensor0D::new(1.0);

        let r = &b - a.trace();
        assert_eq!(r.data(), &0.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &-1.0);
        assert_eq!(gradients.ref_gradient(&b), &1.0);
    }

    #[test]
    fn test_mul_0d() {
        let a = Tensor0D::new(2.0);
        let b = Tensor0D::new(3.0);

        let r = &b * a.trace();
        assert_eq!(r.data(), &6.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &3.0);
        assert_eq!(gradients.ref_gradient(&b), &2.0);
    }

    #[test]
    fn test_div_0d() {
        let a = Tensor0D::new(2.0);
        let b = Tensor0D::new(4.0);

        let r = &b / a.trace();
        assert_eq!(r.data(), &2.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &-1.0);
        assert_eq!(gradients.ref_gradient(&b), &0.5);
    }

    #[test]
    fn test_add_1d() {
        let a = Tensor1D::new([1.0, 2.0, 3.0]);
        let b = Tensor1D::new([1.0, -1.0, 0.0]);

        let r = &b + a.trace();
        assert_eq!(r.data(), &[2.0, 1.0, 3.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &[1.0 / 3.0; 3]);
    }

    #[test]
    fn test_sub_1d() {
        let a = Tensor1D::new([1.0, 2.0, 3.0]);
        let b = Tensor1D::new([1.0, -1.0, 0.0]);

        let r = &b - a.trace();
        assert_eq!(r.data(), &[0.0, -3.0, -3.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[-1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &[1.0 / 3.0; 3]);
    }

    #[test]
    fn test_mul_1d() {
        let a = Tensor1D::new([1.0, 2.0, 3.0]);
        let b = Tensor1D::new([1.0, -1.0, 0.0]);

        let r = &b * a.trace();
        assert_eq!(r.data(), &[1.0, -2.0, 0.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0, -1.0 / 3.0, 0.0]);
        assert_eq!(gradients.ref_gradient(&b), &[1.0 / 3.0, 2.0 / 3.0, 1.0]);
    }

    #[test]
    fn test_div_1d() {
        let a = Tensor1D::new([1.0, 2.0, 3.0]);
        let b = Tensor1D::new([1.0, -1.0, 0.0]);

        let r = &b / a.trace();
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

        let r = &b + a.trace();
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

        let r = &b - a.trace();
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

        let r = &b * a.trace();
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

        let r = &b / a.trace();
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
}
