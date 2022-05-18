mod fill;
mod map;
mod map_inner;
mod reduce;
mod reduce_inner;
mod zero;
mod zip_map;

pub struct Cpu;

pub use fill::*;
pub use map::*;
pub use map_inner::*;
pub use reduce::*;
pub use reduce_inner::*;
pub use zero::*;
pub use zip_map::*;

pub trait Device<T: crate::arrays::CountElements>:
    FillElements<T> + ZipMapElements<T, T> + MapElements<T> + ReduceElements<T> + AllocateZeros
{
}

impl Device<f32> for Cpu {}
impl<T: crate::arrays::CountElements, const M: usize> Device<[T; M]> for Cpu where Cpu: Device<T> {}
