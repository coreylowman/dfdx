use crate::{
    arrays::{Dtype, Shape},
    devices::{
        device::{HasErr, UnaryKernel},
        unary_ops, Device,
    },
    gradients::Tape,
    tensor::Tensor,
};

use super::utils::try_unary_op;

/// `t + val`. `val` is used for all elements of `t`.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([1.0, 2.0, -3.0]);
/// let r = t + 0.5;
/// assert_eq!(r.data(), &[1.5, 2.5, -2.5]);
/// ```
pub trait TryAddScalar<E: Dtype>: HasErr {
    fn add_scalar(self, scalar: E) -> Self {
        self.try_add_scalar(scalar).unwrap()
    }
    fn try_add_scalar(self, scalar: E) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> TryAddScalar<E> for Tensor<S, E, D, T>
where
    D: UnaryKernel<unary_ops::ScalarAdd<E>, S, S, E>,
{
    fn try_add_scalar(self, s: E) -> Result<Self, Self::Err> {
        let op = unary_ops::ScalarAdd(s);
        try_unary_op(op, self)
    }
}

/// `t - val`. `val` is used for all elements of `t`.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([1.0, 2.0, -3.0]);
/// let r = t - 0.5;
/// assert_eq!(r.data(), &[0.5, 1.5, -3.5]);
/// ```
pub trait TrySubScalar<E: Dtype>: HasErr {
    fn sub_scalar(self, scalar: E) -> Self {
        self.try_sub_scalar(scalar).unwrap()
    }
    fn try_sub_scalar(self, scalar: E) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> TrySubScalar<E> for Tensor<S, E, D, T>
where
    D: UnaryKernel<unary_ops::ScalarSub<E>, S, S, E>,
{
    fn try_sub_scalar(self, s: E) -> Result<Self, Self::Err> {
        let op = unary_ops::ScalarSub(s);
        try_unary_op(op, self)
    }
}

pub trait TryMulScalar<E: Dtype>: HasErr {
    fn mul_scalar(self, scalar: E) -> Self {
        self.try_mul_scalar(scalar).unwrap()
    }
    fn try_mul_scalar(self, scalar: E) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> TryMulScalar<E> for Tensor<S, E, D, T>
where
    D: UnaryKernel<unary_ops::ScalarMul<E>, S, S, E>,
{
    fn try_mul_scalar(self, s: E) -> Result<Self, Self::Err> {
        let op = unary_ops::ScalarMul(s);
        try_unary_op(op, self)
    }
}

pub trait TryDivScalar<E: Dtype>: HasErr {
    fn div_scalar(self, scalar: E) -> Self {
        self.try_div_scalar(scalar).unwrap()
    }
    fn try_div_scalar(self, scalar: E) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> TryDivScalar<E> for Tensor<S, E, D, T>
where
    D: UnaryKernel<unary_ops::ScalarDiv<E>, S, S, E>,
{
    fn try_div_scalar(self, s: E) -> Result<Self, Self::Err> {
        let op = unary_ops::ScalarDiv(s);
        try_unary_op(op, self)
    }
}

// use super::utils::move_tape_and_add_backward_op;
// use crate::gradients::Tape;
// use crate::{
//     devices::{Device, ForEachElement},
//     prelude::*,
// };
// use std::ops::{Add, Div, Mul, Sub};

// /// `t + val`. `val` is used for all elements of `t`.
// ///
// /// Example:
// /// ```rust
// /// # use dfdx::prelude::*;
// /// let t = Tensor1D::new([1.0, 2.0, -3.0]);
// /// let r = t + 0.5;
// /// assert_eq!(r.data(), &[1.5, 2.5, -2.5]);
// /// ```
// pub fn add_scalar<T: Tensor<Dtype = f32>>(t: T, val: T::Dtype) -> T {
//     let result = T::NoTape::new_boxed(T::Device::map(t.data(), |x| x + val));
//     move_tape_and_add_backward_op(t, result, move |t, result, grads| {
//         let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
//         T::Device::foreach_mr(t_grad, result_grad, &mut |t, r| {
//             *t += r;
//         });
//     })
// }

// /// `t - val`. `val` is used for all elements of `t`.
// ///
// /// Example:
// /// ```rust
// /// # use dfdx::prelude::*;
// /// let t = Tensor1D::new([1.0, 2.0, -3.0]);
// /// let r = t - 0.5;
// /// assert_eq!(r.data(), &[0.5, 1.5, -3.5]);
// /// ```
// pub fn sub_scalar<T: Tensor<Dtype = f32>>(t: T, val: T::Dtype) -> T {
//     let result = T::NoTape::new_boxed(T::Device::map(t.data(), |x| x - val));
//     move_tape_and_add_backward_op(t, result, move |t, result, grads| {
//         let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
//         T::Device::foreach_mr(t_grad, result_grad, &mut |t, r| {
//             *t += r;
//         });
//     })
// }

// /// `t * val`. `val` is used for all elements of `t`.
// ///
// /// Example:
// /// ```rust
// /// # use dfdx::prelude::*;
// /// let t = Tensor1D::new([1.0, 2.0, -3.0]);
// /// let r = t * 0.5;
// /// assert_eq!(r.data(), &[0.5, 1.0, -1.5]);
// /// ```
// pub fn mul_scalar<T: Tensor<Dtype = f32>>(t: T, val: T::Dtype) -> T {
//     let result = T::NoTape::new_boxed(T::Device::map(t.data(), |x| x * val));
//     move_tape_and_add_backward_op(t, result, move |t, result, grads| {
//         let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
//         T::Device::foreach_mr(t_grad, result_grad, &mut |t, r| {
//             *t += r * val;
//         });
//     })
// }

// /// `t / val`. `val` is used for all elements of `t`.
// ///
// /// Example:
// /// ```rust
// /// # use dfdx::prelude::*;
// /// let t = Tensor1D::new([1.0, 2.0, -3.0]);
// /// let r = t / 2.0;
// /// assert_eq!(r.data(), &[0.5, 1.0, -1.5]);
// /// ```
// pub fn div_scalar<T: Tensor<Dtype = f32>>(t: T, val: T::Dtype) -> T {
//     let result = T::NoTape::new_boxed(T::Device::map(t.data(), |x| x / val));
//     move_tape_and_add_backward_op(t, result, move |t, result, grads| {
//         let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
//         T::Device::foreach_mr(t_grad, result_grad, &mut |t, r| {
//             *t += r / val;
//         });
//     })
// }

// macro_rules! scalar_ops_impl {
//     ($typename:ident, [$($Vs:tt),*]) => {
// impl<$(const $Vs: usize, )* H: Tape> Add<f32> for $typename<$($Vs, )* H> {
//     type Output = Self;
//     /// Calls [add_scalar()] - implements `T<H> + f32`
//     fn add(self, rhs: f32) -> Self::Output {
//         add_scalar(self, rhs)
//     }
// }

// impl<$(const $Vs: usize, )* H: Tape> Add<$typename<$($Vs, )* H>> for f32 {
//     type Output = $typename<$($Vs, )* H>;
//     /// Calls [add_scalar()] - implements `f32 + T<H>`
//     fn add(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
//         add_scalar(rhs, self)
//     }
// }

// impl<$(const $Vs: usize, )* H: Tape> Sub<f32> for $typename<$($Vs, )* H> {
//     type Output = Self;
//     /// Calls [sub_scalar()] - implements `T<H> - f32`
//     fn sub(self, rhs: f32) -> Self::Output {
//         sub_scalar(self, rhs)
//     }
// }

// impl<$(const $Vs: usize, )* H: Tape> Sub<$typename<$($Vs, )* H>> for f32 {
//     type Output = $typename<$($Vs, )* H>;
//     /// Calls [add_scalar()] with neg(rhs) - implements `-T<H> + f32`
//     fn sub(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
//         add_scalar(-rhs, self)
//     }
// }

// impl<$(const $Vs: usize, )* H: Tape> Mul<f32> for $typename<$($Vs, )* H> {
//     type Output = Self;
//     /// Calls [mul_scalar()] - implements `T<H> * f32`
//     fn mul(self, rhs: f32) -> Self::Output {
//         mul_scalar(self, rhs)
//     }
// }

// impl<$(const $Vs: usize, )* H: Tape> Mul<$typename<$($Vs, )* H>> for f32 {
//     type Output = $typename<$($Vs, )* H>;
//     /// Calls [mul_scalar()] - implements `f32 * T<H>`
//     fn mul(self, rhs: $typename<$($Vs, )* H>) -> Self::Output {
//         mul_scalar(rhs, self)
//     }
// }

// impl<$(const $Vs: usize, )* H: Tape> Div<f32> for $typename<$($Vs, )* H> {
//     type Output = Self;
//     /// Calls [div_scalar()] - implements `T<H> / f32`
//     fn div(self, rhs: f32) -> Self::Output {
//         div_scalar(self, rhs)
//     }
// }
//     };
// }

// scalar_ops_impl!(Tensor0D, []);
// scalar_ops_impl!(Tensor1D, [N]);
// scalar_ops_impl!(Tensor2D, [M, N]);
// scalar_ops_impl!(Tensor3D, [M, N, O]);
// scalar_ops_impl!(Tensor4D, [M, N, O, P]);

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_scalar_add_0d() {
//         let x = tensor(0.0);
//         let r = x.trace() + 1.0;
//         assert_eq!(r.data(), &1.0);
//         let gradients = backward(r.exp());
//         assert_eq!(gradients.ref_gradient(&x), &1.0f32.exp());
//     }

//     #[test]
//     fn test_scalar_add_1d() {
//         let x = tensor([0.0, 1.0, 2.0]);
//         let r = x.trace() + 0.5;
//         assert_eq!(r.data(), &[0.5, 1.5, 2.5]);
//         let gradients = backward(r.exp().sum());
//         assert_eq!(
//             gradients.ref_gradient(&x),
//             &[1.6487212, 4.481689, 12.182494]
//         );
//     }

//     #[test]
//     fn test_scalar_add_2d() {
//         let x = Tensor2D::zeros();
//         let r = x.trace() + 0.5;
//         assert_eq!(r.data(), &[[0.5; 2]; 3]);
//         let gradients = backward(r.exp().sum());
//         assert_eq!(gradients.ref_gradient(&x), &[[1.6487212; 2]; 3]);
//     }

//     #[test]
//     fn test_scalar_sub_0d() {
//         let x = tensor(0.0);
//         let r = x.trace() - 1.0;
//         assert_eq!(r.data(), &-1.0);
//         let gradients = backward(r.exp().sum());
//         assert_eq!(gradients.ref_gradient(&x), &(-1.0f32).exp());
//     }

//     #[test]
//     fn test_scalar_sub_1d() {
//         let x = tensor([0.0, 1.0, 2.0]);
//         let r = x.trace() - 1.0;
//         assert_eq!(r.data(), &[-1.0, 0.0, 1.0]);
//         let gradients = backward(r.exp().sum());
//         assert_eq!(gradients.ref_gradient(&x), &[0.36787945, 1.0, 2.7182817]);
//     }

//     #[test]
//     fn test_scalar_sub_2d() {
//         let x = Tensor2D::zeros();
//         let r = x.trace() - 1.0;
//         assert_eq!(r.data(), &[[-1.0; 2]; 3]);
//         let gradients = backward(r.exp().sum());
//         assert_eq!(gradients.ref_gradient(&x), &[[0.36787945; 2]; 3]);
//     }

//     #[test]
//     fn test_scalar_mul_0d() {
//         let x = tensor(1.0);
//         let r = x.trace() * 0.5;
//         assert_eq!(r.data(), &0.5);
//         let gradients = backward(r.exp().sum());
//         assert_eq!(gradients.ref_gradient(&x), &0.8243606);
//     }

//     #[test]
//     fn test_scalar_mul_1d() {
//         let x = tensor([0.0, 1.0, 2.0]);
//         let r = x.trace() * 0.5;
//         assert_eq!(r.data(), &[0.0, 0.5, 1.0]);
//         let gradients = backward(r.exp().sum());
//         assert_eq!(gradients.ref_gradient(&x), &[0.5, 0.8243606, 1.3591409]);
//     }

//     #[test]
//     fn test_scalar_mul_2d() {
//         let x = Tensor2D::ones();
//         let r = x.trace() * 0.5;
//         assert_eq!(r.data(), &[[0.5; 2]; 3]);
//         let gradients = backward(r.exp().sum());
//         assert_eq!(gradients.ref_gradient(&x), &[[0.8243606; 2]; 3]);
//     }

//     #[test]
//     fn test_scalar_div_0d() {
//         let x = tensor(1.0);
//         let r = x.trace() / 2.0;
//         assert_eq!(r.data(), &0.5);
//         let gradients = backward(r.exp().sum());
//         assert_eq!(gradients.ref_gradient(&x), &0.8243606);
//     }

//     #[test]
//     fn test_scalar_div_1d() {
//         let x = tensor([0.0, 1.0, 2.0]);
//         let r = x.trace() / 2.0;
//         assert_eq!(r.data(), &[0.0, 0.5, 1.0]);
//         let gradients = backward(r.exp().sum());
//         assert_eq!(gradients.ref_gradient(&x), &[0.5, 0.8243606, 1.3591409]);
//     }

//     #[test]
//     fn test_scalar_div_2d() {
//         let x = Tensor2D::ones();
//         let r = x.trace() / 2.0;
//         assert_eq!(r.data(), &[[0.5; 2]; 3]);
//         let gradients = backward(r.exp().sum());
//         assert_eq!(gradients.ref_gradient(&x), &[[0.8243606; 2]; 3]);
//     }
// }
