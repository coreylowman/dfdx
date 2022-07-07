use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

/// Adds `val` to all elements of `t`.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([1.0, 2.0, -3.0]);
/// let r = t + 0.5;
/// assert_eq!(r.data(), &[1.5, 2.5, -2.5]);
/// ```
pub fn scalar_add<T: Tensor<Dtype = f32>>(t: T, val: T::Dtype) -> T {
    let result = T::NoTape::new_boxed(T::Device::map(t.data(), |x| x + val));
    move_tape_and_add_backward_op(t, result, move |t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        T::Device::foreach_mr(t_grad, result_grad, &mut |t, r| {
            *t += r;
        });
    })
}

/// Subtracts `val` from all elements of `t`.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([1.0, 2.0, -3.0]);
/// let r = t - 0.5;
/// assert_eq!(r.data(), &[0.5, 1.5, -3.5]);
/// ```
pub fn scalar_sub<T: Tensor<Dtype = f32>>(t: T, val: T::Dtype) -> T {
    let result = T::NoTape::new_boxed(T::Device::map(t.data(), |x| x - val));
    move_tape_and_add_backward_op(t, result, move |t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        T::Device::foreach_mr(t_grad, result_grad, &mut |t, r| {
            *t += r;
        });
    })
}

/// Multiplies all elements of `t` by `val`.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([1.0, 2.0, -3.0]);
/// let r = t * 0.5;
/// assert_eq!(r.data(), &[0.5, 1.0, -1.5]);
/// ```
pub fn scalar_mul<T: Tensor<Dtype = f32>>(t: T, val: T::Dtype) -> T {
    let result = T::NoTape::new_boxed(T::Device::map(t.data(), |x| x * val));
    move_tape_and_add_backward_op(t, result, move |t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        T::Device::foreach_mr(t_grad, result_grad, &mut |t, r| {
            *t += r * val;
        });
    })
}

/// Divides all elements of `t` by `val`.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([1.0, 2.0, -3.0]);
/// let r = t / 2.0;
/// assert_eq!(r.data(), &[0.5, 1.0, -1.5]);
/// ```
pub fn scalar_div<T: Tensor<Dtype = f32>>(t: T, val: T::Dtype) -> T {
    let result = T::NoTape::new_boxed(T::Device::map(t.data(), |x| x / val));
    move_tape_and_add_backward_op(t, result, move |t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        T::Device::foreach_mr(t_grad, result_grad, &mut |t, r| {
            *t += r / val;
        });
    })
}

macro_rules! scalar_ops_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> Add<f32> for $typename<$($Vs, )* H> {
    type Output = Self;
    /// Calls [scalar_add()] - implements `T<H> + f32`
    fn add(self, rhs: f32) -> Self::Output {
        scalar_add(self, rhs)
    }
}

impl<$(const $Vs: usize, )* H: Tape> Add<$typename<$($Vs, )* H>> for f32 {
    type Output = $typename<$($Vs, )* H>;
    /// Calls [scalar_add()] - implements `f32 + T<H>`
    fn add(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
        scalar_add(rhs, self)
    }
}

impl<$(const $Vs: usize, )* H: Tape> Sub<f32> for $typename<$($Vs, )* H> {
    type Output = Self;
    /// Calls [scalar_sub()] - implements `T<H> - f32`
    fn sub(self, rhs: f32) -> Self::Output {
        scalar_sub(self, rhs)
    }
}

impl<$(const $Vs: usize, )* H: Tape> Sub<$typename<$($Vs, )* H>> for f32 {
    type Output = $typename<$($Vs, )* H>;
    /// Calls [scalar_add()] with neg(rhs) - implements `-T<H> + f32`
    fn sub(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
        scalar_add(-rhs, self)
    }
}

impl<$(const $Vs: usize, )* H: Tape> Mul<f32> for $typename<$($Vs, )* H> {
    type Output = Self;
    /// Calls [scalar_mul()] - implements `T<H> * f32`
    fn mul(self, rhs: f32) -> Self::Output {
        scalar_mul(self, rhs)
    }
}

impl<$(const $Vs: usize, )* H: Tape> Mul<$typename<$($Vs, )* H>> for f32 {
    type Output = $typename<$($Vs, )* H>;
    /// Calls [scalar_mul()] - implements `f32 * T<H>`
    fn mul(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
        scalar_mul(rhs, self)
    }
}

impl<$(const $Vs: usize, )* H: Tape> Div<f32> for $typename<$($Vs, )* H> {
    type Output = Self;
    /// Calls [scalar_div()] - implements `T<H> / f32`
    fn div(self, rhs: f32) -> Self::Output {
        scalar_div(self, rhs)
    }
}
    };
}

scalar_ops_impl!(Tensor0D, []);
scalar_ops_impl!(Tensor1D, [N]);
scalar_ops_impl!(Tensor2D, [M, N]);
scalar_ops_impl!(Tensor3D, [M, N, O]);
scalar_ops_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_add_0d() {
        let x = Tensor0D::new(0.0);
        let r = x.trace() + 1.0;
        assert_eq!(r.data(), &1.0);
        let gradients = r.exp().backward();
        assert_eq!(gradients.ref_gradient(&x), &1.0f32.exp());
    }

    #[test]
    fn test_scalar_add_1d() {
        let x = Tensor1D::new([0.0, 1.0, 2.0]);
        let r = x.trace() + 0.5;
        assert_eq!(r.data(), &[0.5, 1.5, 2.5]);
        let gradients = r.exp().sum().backward();
        assert_eq!(
            gradients.ref_gradient(&x),
            &[1.6487212, 4.481689, 12.182494]
        );
    }

    #[test]
    fn test_scalar_add_2d() {
        let x = Tensor2D::zeros();
        let r = x.trace() + 0.5;
        assert_eq!(r.data(), &[[0.5; 2]; 3]);
        let gradients = r.exp().sum().backward();
        assert_eq!(gradients.ref_gradient(&x), &[[1.6487212; 2]; 3]);
    }

    #[test]
    fn test_scalar_sub_0d() {
        let x = Tensor0D::new(0.0);
        let r = x.trace() - 1.0;
        assert_eq!(r.data(), &-1.0);
        let gradients = r.exp().sum().backward();
        assert_eq!(gradients.ref_gradient(&x), &(-1.0f32).exp());
    }

    #[test]
    fn test_scalar_sub_1d() {
        let x = Tensor1D::new([0.0, 1.0, 2.0]);
        let r = x.trace() - 1.0;
        assert_eq!(r.data(), &[-1.0, 0.0, 1.0]);
        let gradients = r.exp().sum().backward();
        assert_eq!(gradients.ref_gradient(&x), &[0.36787945, 1.0, 2.7182817]);
    }

    #[test]
    fn test_scalar_sub_2d() {
        let x = Tensor2D::zeros();
        let r = x.trace() - 1.0;
        assert_eq!(r.data(), &[[-1.0; 2]; 3]);
        let gradients = r.exp().sum().backward();
        assert_eq!(gradients.ref_gradient(&x), &[[0.36787945; 2]; 3]);
    }

    #[test]
    fn test_scalar_mul_0d() {
        let x = Tensor0D::new(1.0);
        let r = x.trace() * 0.5;
        assert_eq!(r.data(), &0.5);
        let gradients = r.exp().sum().backward();
        assert_eq!(gradients.ref_gradient(&x), &0.8243606);
    }

    #[test]
    fn test_scalar_mul_1d() {
        let x = Tensor1D::new([0.0, 1.0, 2.0]);
        let r = x.trace() * 0.5;
        assert_eq!(r.data(), &[0.0, 0.5, 1.0]);
        let gradients = r.exp().sum().backward();
        assert_eq!(gradients.ref_gradient(&x), &[0.5, 0.8243606, 1.3591409]);
    }

    #[test]
    fn test_scalar_mul_2d() {
        let x = Tensor2D::ones();
        let r = x.trace() * 0.5;
        assert_eq!(r.data(), &[[0.5; 2]; 3]);
        let gradients = r.exp().sum().backward();
        assert_eq!(gradients.ref_gradient(&x), &[[0.8243606; 2]; 3]);
    }

    #[test]
    fn test_scalar_div_0d() {
        let x = Tensor0D::new(1.0);
        let r = x.trace() / 2.0;
        assert_eq!(r.data(), &0.5);
        let gradients = r.exp().sum().backward();
        assert_eq!(gradients.ref_gradient(&x), &0.8243606);
    }

    #[test]
    fn test_scalar_div_1d() {
        let x = Tensor1D::new([0.0, 1.0, 2.0]);
        let r = x.trace() / 2.0;
        assert_eq!(r.data(), &[0.0, 0.5, 1.0]);
        let gradients = r.exp().sum().backward();
        assert_eq!(gradients.ref_gradient(&x), &[0.5, 0.8243606, 1.3591409]);
    }

    #[test]
    fn test_scalar_div_2d() {
        let x = Tensor2D::ones();
        let r = x.trace() / 2.0;
        assert_eq!(r.data(), &[[0.5; 2]; 3]);
        let gradients = r.exp().sum().backward();
        assert_eq!(gradients.ref_gradient(&x), &[[0.8243606; 2]; 3]);
    }
}
