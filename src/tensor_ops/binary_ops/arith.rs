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

pub fn apply_binary<T: Tensor, F: DiffBinaryFunction<f32>>(lhs: &T::NoTape, rhs: T) -> T {
    let result = T::NoTape::new_boxed(T::Device::zip_map(lhs.data(), rhs.data(), F::f));
    let mut lhs_deriv = T::Device::zip_map(lhs.data(), rhs.data(), F::dfdx);
    let mut rhs_deriv = T::Device::zip_map(lhs.data(), rhs.data(), F::dfdy);
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

// TODO how to combine this with above?
pub fn apply_binary_lhs<T: Tensor, F: DiffBinaryFunction<f32>>(lhs: T, rhs: &T::NoTape) -> T {
    let result = T::NoTape::new_boxed(T::Device::zip_map(lhs.data(), rhs.data(), F::f));
    let mut lhs_deriv = T::Device::zip_map(lhs.data(), rhs.data(), F::dfdx);
    let mut rhs_deriv = T::Device::zip_map(lhs.data(), rhs.data(), F::dfdy);
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let result_grad = tape.ref_gradient(&_result);
        T::Device::mul_assign(lhs_deriv.as_mut(), result_grad);
        T::Device::mul_assign(rhs_deriv.as_mut(), result_grad);
        T::Device::add_assign(tape.mut_gradient(&lhs), lhs_deriv.as_ref());
        T::Device::add_assign(tape.mut_gradient(&_rhs), rhs_deriv.as_ref());
    });
    result.with_tape_holder(tape_holder)
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
        assert_eq!(gradients.ref_gradient(&a), &1.0);
        assert_eq!(gradients.ref_gradient(&b), &1.0);

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

        let r = a.trace() - &b;
        assert_eq!(r.data(), &0.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &1.0);
        assert_eq!(gradients.ref_gradient(&b), &-1.0);

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

        let r = mul_lhs(a.trace(), &b);
        assert_eq!(r.data(), &6.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &3.0);
        assert_eq!(gradients.ref_gradient(&b), &2.0);
    }

    #[test]
    fn test_div_0d() {
        let a = Tensor0D::new(2.0);
        let b = Tensor0D::new(4.0);

        let r = a.trace() / &b;
        assert_eq!(r.data(), &0.5);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &0.25);
        assert_eq!(gradients.ref_gradient(&b), &-0.125);

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

        let r = a.trace() + &b;
        assert_eq!(r.data(), &[2.0, 1.0, 3.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &[1.0 / 3.0; 3]);

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

        let r = a.trace() - &b;
        assert_eq!(r.data(), &[0.0, 3.0, 3.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &[-1.0 / 3.0; 3]);

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

        let r = mul_lhs(a.trace(), &b);
        assert_eq!(r.data(), &[1.0, -2.0, 0.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0, -1.0 / 3.0, 0.0]);
        assert_eq!(gradients.ref_gradient(&b), &[1.0 / 3.0, 2.0 / 3.0, 1.0]);

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

        let r = a.trace() / &b;
        assert_eq!(r.data(), &[1.0, -2.0, f32::INFINITY]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[1.0 / 3.0, -1.0 / 3.0, f32::INFINITY]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[-1.0 / 3.0, -2.0 / 3.0, f32::NEG_INFINITY]
        );

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

        let r = a.trace() + &b;
        assert_eq!(
            r.data(),
            &[[1.1769, 0.5552, 0.5259], [1.3917, 1.0692, 0.873]]
        );
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[1.0 / 6.0; 3]; 2]);
        assert_eq!(gradients.ref_gradient(&b), &[[1.0 / 6.0; 3]; 2]);

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

        let r = a.trace() - &b;
        assert_eq!(
            r.data(),
            &[
                [0.13709998, -0.21360001, -0.2259],
                [-0.2601, 0.33279997, 0.7954]
            ]
        );
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[1.0 / 6.0; 3]; 2]);
        assert_eq!(gradients.ref_gradient(&b), &[[-1.0 / 6.0; 3]; 2]);

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

        let r = mul_lhs(a.trace(), &b);
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

        let r = a.trace() / &b;
        assert_eq!(
            r.data(),
            &[
                [1.2637045, 0.44432878, 0.3990423],
                [0.6850708, 1.9038565, 21.5]
            ]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [0.32057446, 0.4335761, 0.44338033],
                [0.20180005, 0.45265254, 4.2955327]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[
                [-0.40511137, -0.19265036, -0.1769275],
                [-0.13824734, -0.86178553, -92.35394]
            ]
        );

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
