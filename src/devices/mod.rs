//! Provides implementations for modifying Nd arrays on the [Cpu].

mod fill;
mod foreach;
mod gather;
mod map;
mod reduce;
mod reduce_last_dim;
mod zero;
mod zip_map;

/// The CPU device
pub struct Cpu;

pub use fill::*;
pub use foreach::*;
pub use gather::*;
pub use map::*;
pub use reduce::*;
pub use reduce_last_dim::*;
pub use zero::*;
pub use zip_map::*;

/// Represents something that can act on `T`.
pub trait Device<T: crate::arrays::CountElements>:
    FillElements<T>
    + ZipMapElements<T, T>
    + MapElements<T>
    + ReduceElements<T>
    + AllocateZeros
    + ReduceLastDim<T>
    + ZipMapElements<T, <Self as ReduceLastDim<T>>::Reduced>
    + GatherElements<T>
    + ForEachElement<T>
{
}

impl Device<f32> for Cpu {}
impl<T: crate::arrays::CountElements, const M: usize> Device<[T; M]> for Cpu where
    Cpu: Device<T>
        + ReduceLastDim<[T; M]>
        + ZipMapElements<[T; M], <Self as ReduceLastDim<[T; M]>>::Reduced>
        + GatherElements<[T; M]>
{
}

/// A [HasArrayType] that has a [Device] for its [HasArrayType::Array]
pub trait HasDevice: crate::arrays::HasArrayType {
    type Device: Device<Self::Array>;
}
