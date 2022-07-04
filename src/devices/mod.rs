//! Provides implementations for modifying Nd arrays on the [Cpu].

mod allocate;
mod fill;
mod foreach;
mod reduce;
mod reduce_last_dim;

/// The CPU device
pub struct Cpu;

pub use allocate::*;
pub use fill::*;
pub use foreach::*;
pub use reduce::*;
pub use reduce_last_dim::*;

/// Represents something that can act on `T`.
pub trait Device<T: crate::arrays::CountElements>:
    FillElements<T>
    + ReduceElements<T>
    + AllocateZeros
    + ReduceLastDim<T>
    + ForEachElement<T>
    + BroadcastForEach<T, <Self as ReduceLastDim<T>>::Reduced>
{
    fn map<F: FnMut(&T::Dtype) -> T::Dtype>(t: &T, mut f: F) -> Box<T> {
        let mut out: Box<T> = Self::zeros();
        Self::foreach_mr(out.as_mut(), t, &mut |o, t| *o = f(t));
        out
    }

    fn add(lhs: &mut T, rhs: &T)
    where
        T::Dtype: std::ops::AddAssign + Copy,
    {
        Self::foreach_mr(lhs, rhs, &mut |l, r| {
            *l += *r;
        })
    }

    fn badd(lhs: &mut T, rhs: Broadcast<<Self as ReduceLastDim<T>>::Reduced>)
    where
        T::Dtype: std::ops::AddAssign + Copy,
    {
        Self::foreach_mb(lhs, rhs, &mut |l, r| {
            *l += *r;
        })
    }

    fn sub(lhs: &mut T, rhs: &T)
    where
        T::Dtype: std::ops::SubAssign + Copy,
    {
        Self::foreach_mr(lhs, rhs, &mut |l, r| {
            *l -= *r;
        })
    }

    fn bsub(lhs: &mut T, rhs: Broadcast<<Self as ReduceLastDim<T>>::Reduced>)
    where
        T::Dtype: std::ops::SubAssign + Copy,
    {
        Self::foreach_mb(lhs, rhs, &mut |l, r| {
            *l -= *r;
        })
    }

    fn addmul(out: &mut T, lhs: &T, rhs: &T)
    where
        T::Dtype: std::ops::Mul<Output = T::Dtype> + std::ops::AddAssign + Copy,
    {
        Self::foreach_mrr(out, lhs, rhs, &mut |o, l, r| {
            *o += *l * *r;
        })
    }
}

impl Device<f32> for Cpu {}
impl<T: crate::arrays::CountElements, const M: usize> Device<[T; M]> for Cpu where
    Cpu: Device<T>
        + ReduceLastDim<[T; M]>
        + BroadcastForEach<[T; M], <Self as ReduceLastDim<[T; M]>>::Reduced>
{
}

/// A [HasArrayType] that has a [Device] for its [HasArrayType::Array]
pub trait HasDevice: crate::arrays::HasArrayType {
    type Device: Device<Self::Array>;
}
