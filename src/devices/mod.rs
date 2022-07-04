//! Provides implementations for modifying Nd arrays on the [Cpu].

mod allocate;
mod fill;
mod foreach;
mod map;
mod reduce;
mod reduce_last_dim;

/// The CPU device
pub struct Cpu;

pub use allocate::*;
pub use fill::*;
pub use foreach::*;
pub use map::*;
pub use reduce::*;
pub use reduce_last_dim::*;

/// Represents something that can act on `T`.
pub trait Device<T: crate::arrays::CountElements>:
    FillElements<T>
    + MapElements<T>
    + ReduceElements<T>
    + AllocateZeros
    + ReduceLastDim<T>
    + ForEachElement<T>
    + BroadcastForEach<T, <Self as ReduceLastDim<T>>::Reduced>
{
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
