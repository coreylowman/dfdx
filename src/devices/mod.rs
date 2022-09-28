//! Provides implementations for modifying Nd arrays on the [Cpu].

mod allocate;
mod broadcast_reduce;
mod fill;
mod foreach;
mod matmul;
mod permute;
mod reduce_all;
mod select;

pub use allocate::*;
pub use broadcast_reduce::*;
pub use fill::*;
pub use foreach::*;
pub use matmul::*;
pub use permute::*;
pub use reduce_all::*;
pub use select::*;

use std::ops::*;

/// The CPU device
pub struct Cpu;

/// Represents something that can act on `T`.
pub trait Device<T: crate::arrays::CountElements>:
    FillElements<T> + ReduceAllElements<T> + AllocateZeros + ForEachElement<T>
{
    /// Allocate a new `T` and then store `f` applied to `t` in the new `T`. Uses [ForEachElement::foreach_mr].
    fn map<F: FnMut(&T::Dtype) -> T::Dtype>(t: &T, mut f: F) -> Box<T> {
        let mut out: Box<T> = Self::zeros();
        Self::foreach_mr(out.as_mut(), t, &mut |o, t| *o = f(t));
        out
    }

    /// Computes `lhs += rhs`, using [ForEachElement::foreach_mr].
    fn add(lhs: &mut T, rhs: &T)
    where
        T::Dtype: for<'r> AddAssign<&'r T::Dtype> + Copy,
    {
        Self::foreach_mr(lhs, rhs, &mut |l, r| l.add_assign(r))
    }

    /// Computes `lhs -= rhs` using [ForEachElement::foreach_mr]
    fn sub(lhs: &mut T, rhs: &T)
    where
        T::Dtype: for<'r> SubAssign<&'r T::Dtype> + Copy,
    {
        Self::foreach_mr(lhs, rhs, &mut |l, r| l.sub_assign(r))
    }

    /// Computes `out += lhs * rhs` using [ForEachElement::foreach_mrr].
    fn addmul(out: &mut T, lhs: &T, rhs: &T)
    where
        T::Dtype: AddAssign,
        for<'r> &'r T::Dtype: Mul<Output = T::Dtype>,
    {
        Self::foreach_mrr(out, lhs, rhs, &mut |o, l, r| o.add_assign(l * r))
    }
}

impl Device<f32> for Cpu {}
impl<const M: usize> Device<[f32; M]> for Cpu {}
impl<const M: usize, const N: usize> Device<[[f32; N]; M]> for Cpu {}
impl<const M: usize, const N: usize, const O: usize> Device<[[[f32; O]; N]; M]> for Cpu {}
impl<const M: usize, const N: usize, const O: usize, const P: usize> Device<[[[[f32; P]; O]; N]; M]>
    for Cpu
{
}

/// A [crate::arrays::HasArrayType] that has a [Device] for its [crate::arrays::HasArrayType::Array]
pub trait HasDevice: crate::arrays::HasArrayType {
    type Device: Device<Self::Array>;
}
