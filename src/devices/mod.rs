//! Provides implementations for modifying Nd arrays on the CPU.

mod fill;
mod map;
mod map_inner;
mod reduce;
mod reduce_inner;
mod zero;
mod zip_map;

/// The CPU device
pub struct Cpu;

pub use fill::*;
pub use map::*;
pub use map_inner::*;
pub use reduce::*;
pub use reduce_inner::*;
pub use zero::*;
pub use zip_map::*;

/// Represents something that can act on `T`.
pub trait Device<T: crate::arrays::CountElements>:
    FillElements<T> + ZipMapElements<T, T> + MapElements<T> + ReduceElements<T> + AllocateZeros
{
}

impl Device<f32> for Cpu {}
impl<T: crate::arrays::CountElements, const M: usize> Device<[T; M]> for Cpu where Cpu: Device<T> {}

/// A [HasArrayType] that has a [Device] for its [HasArrayType::Array]
pub trait HasDevice: crate::arrays::HasArrayType {
    type Device: Device<Self::Array>;
}
