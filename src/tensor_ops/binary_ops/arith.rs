use crate::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

pub fn add<T: Tensor>(lhs: &T::NoTape, rhs: T) -> T {
    apply_binary::<T, BinaryAdd>(lhs, rhs)
}

pub fn add_lhs<T: Tensor>(lhs: T, rhs: &T::NoTape) -> T {
    apply_binary_lhs::<T, BinaryAdd>(lhs, rhs)
}

pub fn sub<T: Tensor>(lhs: &T::NoTape, rhs: T) -> T {
    apply_binary::<T, BinarySub>(lhs, rhs)
}

pub fn sub_lhs<T: Tensor>(lhs: T, rhs: &T::NoTape) -> T {
    apply_binary_lhs::<T, BinarySub>(lhs, rhs)
}

pub fn mul<T: Tensor>(lhs: &T::NoTape, rhs: T) -> T {
    apply_binary::<T, BinaryMul>(lhs, rhs)
}

pub fn mul_lhs<T: Tensor>(lhs: T, rhs: &T::NoTape) -> T {
    apply_binary_lhs::<T, BinaryMul>(lhs, rhs)
}

pub fn div<T: Tensor>(lhs: &T::NoTape, rhs: T) -> T {
    apply_binary::<T, BinaryDiv>(lhs, rhs)
}

pub fn div_lhs<T: Tensor>(lhs: T, rhs: &T::NoTape) -> T {
    apply_binary_lhs::<T, BinaryDiv>(lhs, rhs)
}

pub fn apply_binary<T: Tensor, F: DiffBinaryFunction>(lhs: &T::NoTape, rhs: T) -> T {
    let result = T::NoTape::new(lhs.data().zip_map(rhs.data(), F::f));
    let lhs_deriv = lhs.data().zip_map(rhs.data(), F::dfdx);
    let rhs_deriv = lhs.data().zip_map(rhs.data(), F::dfdy);
    let (rhs, mut tape_holder) = rhs.split_tape_holder();
    let _lhs = lhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let d_grad_lhs = lhs_deriv.mul(tape.gradient(&_result));
        tape.mut_gradient(&_lhs).add_assign(&d_grad_lhs);
        let d_grad_rhs = rhs_deriv.mul(tape.gradient(&_result));
        tape.mut_gradient(&rhs).add_assign(&d_grad_rhs);
    });
    result.with_tape_holder(tape_holder)
}

pub fn apply_binary_lhs<T: Tensor, F: DiffBinaryFunction>(lhs: T, rhs: &T::NoTape) -> T {
    let result = T::NoTape::new(lhs.data().zip_map(rhs.data(), F::f));
    let lhs_deriv = lhs.data().zip_map(rhs.data(), F::dfdx);
    let rhs_deriv = lhs.data().zip_map(rhs.data(), F::dfdy);
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let d_grad_lhs = lhs_deriv.mul(tape.gradient(&_result));
        tape.mut_gradient(&lhs).add_assign(&d_grad_lhs);
        let d_grad_rhs = rhs_deriv.mul(tape.gradient(&_result));
        tape.mut_gradient(&_rhs).add_assign(&d_grad_rhs);
    });
    result.with_tape_holder(tape_holder)
}

pub trait DiffBinaryFunction {
    fn f(x: &f32, y: &f32) -> f32;
    fn dfdx(x: &f32, y: &f32) -> f32;
    fn dfdy(x: &f32, y: &f32) -> f32;
}

struct BinaryAdd;

impl DiffBinaryFunction for BinaryAdd {
    fn f(x: &f32, y: &f32) -> f32 {
        x + y
    }

    fn dfdx(_: &f32, _: &f32) -> f32 {
        1.0
    }

    fn dfdy(_: &f32, _: &f32) -> f32 {
        1.0
    }
}

struct BinarySub;

impl DiffBinaryFunction for BinarySub {
    fn f(x: &f32, y: &f32) -> f32 {
        x - y
    }

    fn dfdx(_: &f32, _: &f32) -> f32 {
        1.0
    }

    fn dfdy(_: &f32, _: &f32) -> f32 {
        -1.0
    }
}

struct BinaryMul;

impl DiffBinaryFunction for BinaryMul {
    fn f(x: &f32, y: &f32) -> f32 {
        x * y
    }

    fn dfdx(_: &f32, y: &f32) -> f32 {
        *y
    }

    fn dfdy(x: &f32, _: &f32) -> f32 {
        *x
    }
}

struct BinaryDiv;

impl DiffBinaryFunction for BinaryDiv {
    fn f(x: &f32, y: &f32) -> f32 {
        x / y
    }

    fn dfdx(_x: &f32, y: &f32) -> f32 {
        1.0 / y
    }

    fn dfdy(x: &f32, y: &f32) -> f32 {
        -x / y.powi(2)
    }
}

macro_rules! binary_ops_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
// &T<NoTape> + T<H>
impl<$(const $Vs: usize, )* H: TapeHolder> Add<$typename<$($Vs, )* H>> for &$typename<$($Vs, )* NoTape> {
    type Output = $typename<$($Vs, )* H>;
    fn add(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
        add(self, rhs)
    }
}

// T<H> + &T<NoTape>
impl<$(const $Vs: usize, )* H: TapeHolder> Add<&$typename<$($Vs, )* NoTape>> for $typename<$($Vs, )* H> {
    type Output = $typename<$($Vs, )* H>;
    fn add(self, rhs: &$typename<$($Vs, )* NoTape>) -> Self::Output {
        add_lhs(self, rhs)
    }
}

// &T<NoTape> - T<H>
impl<$(const $Vs: usize, )* H: TapeHolder> Sub<$typename<$($Vs, )* H>> for &$typename<$($Vs, )* NoTape> {
    type Output = $typename<$($Vs, )* H>;
    fn sub(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
        sub(self, rhs)
    }
}

// T<H> - &T<NoTape>
impl<$(const $Vs: usize, )* H: TapeHolder> Sub<&$typename<$($Vs, )* NoTape>> for $typename<$($Vs, )* H> {
    type Output = $typename<$($Vs, )* H>;
    fn sub(self, rhs: &$typename<$($Vs, )* NoTape>) -> Self::Output {
        sub_lhs(self, rhs)
    }
}

// &T<NoTape> * T<H>
impl<$(const $Vs: usize, )* H: TapeHolder> Mul<$typename<$($Vs, )* H>> for &$typename<$($Vs, )* NoTape> {
    type Output = $typename<$($Vs, )* H>;
    fn mul(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
        mul(self, rhs)
    }
}

// &T<NoTape> / T<H>
impl<$(const $Vs: usize, )* H: TapeHolder> Div<$typename<$($Vs, )* H>> for &$typename<$($Vs, )* NoTape> {
    type Output = $typename<$($Vs, )* H>;
    fn div(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
        div(self, rhs)
    }
}

// T<H> / &T<NoTape>
impl<$(const $Vs: usize, )* H: TapeHolder> Div<&$typename<$($Vs, )* NoTape>> for $typename<$($Vs, )* H> {
    type Output = $typename<$($Vs, )* H>;
    fn div(self, rhs: &$typename<$($Vs, )* NoTape>) -> Self::Output {
        div_lhs(self, rhs)
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
        assert_eq!(gradients.gradient(&a), &1.0);
        assert_eq!(gradients.gradient(&b), &1.0);

        let r = &b + a.trace();
        assert_eq!(r.data(), &2.0);
        let gradients = r.backward();
        assert_eq!(gradients.gradient(&a), &1.0);
        assert_eq!(gradients.gradient(&b), &1.0);
    }

    #[test]
    fn test_sub_0d() {
        let a = Tensor0D::new(1.0);
        let b = Tensor0D::new(1.0);

        let r = a.trace() - &b;
        assert_eq!(r.data(), &0.0);
        let gradients = r.backward();
        assert_eq!(gradients.gradient(&a), &1.0);
        assert_eq!(gradients.gradient(&b), &-1.0);

        let r = &b - a.trace();
        assert_eq!(r.data(), &0.0);
        let gradients = r.backward();
        assert_eq!(gradients.gradient(&a), &-1.0);
        assert_eq!(gradients.gradient(&b), &1.0);
    }

    #[test]
    fn test_mul_0d() {
        let a = Tensor0D::new(2.0);
        let b = Tensor0D::new(3.0);

        let r = &b * a.trace();
        assert_eq!(r.data(), &6.0);
        let gradients = r.backward();
        assert_eq!(gradients.gradient(&a), &3.0);
        assert_eq!(gradients.gradient(&b), &2.0);

        let r = mul_lhs(a.trace(), &b);
        assert_eq!(r.data(), &6.0);
        let gradients = r.backward();
        assert_eq!(gradients.gradient(&a), &3.0);
        assert_eq!(gradients.gradient(&b), &2.0);
    }

    #[test]
    fn test_div_0d() {
        let a = Tensor0D::new(2.0);
        let b = Tensor0D::new(4.0);

        let r = a.trace() / &b;
        assert_eq!(r.data(), &0.5);
        let gradients = r.backward();
        assert_eq!(gradients.gradient(&a), &0.25);
        assert_eq!(gradients.gradient(&b), &-0.125);

        let r = &b / a.trace();
        assert_eq!(r.data(), &2.0);
        let gradients = r.backward();
        assert_eq!(gradients.gradient(&a), &-1.0);
        assert_eq!(gradients.gradient(&b), &0.5);
    }

    #[test]
    fn test_add_1d() {
        todo!();
    }

    #[test]
    fn test_sub_1d() {
        todo!();
    }

    #[test]
    fn test_mul_1d() {
        todo!();
    }

    #[test]
    fn test_div_1d() {
        todo!();
    }

    #[test]
    fn test_add_2d() {
        todo!();
    }

    #[test]
    fn test_sub_2d() {
        todo!();
    }

    #[test]
    fn test_mul_2d() {
        todo!();
    }

    #[test]
    fn test_div_2d() {
        todo!();
    }
}
