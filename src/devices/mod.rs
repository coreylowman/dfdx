mod cpu_impls;

pub use cpu_impls::*;

pub trait Device<T: crate::arrays::CountElements>:
    FillElements<T>
    + ZipMapElements<T, T, Output = T>
    + MapElements<T>
    + ReduceElements<T>
    + AllocateZeros<T>
{
}

impl Device<f32> for Cpu {}
impl<T: crate::arrays::CountElements, const M: usize> Device<[T; M]> for Cpu where Cpu: Device<T> {}
