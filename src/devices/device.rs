use super::{
    cpu_impls::Cpu, AllocateZeros, CountElements, FillElements, MapElements, ReduceElements,
    ZipMapElements,
};

pub trait Device<T: CountElements>:
    FillElements<T>
    + ZipMapElements<T, T, Output = T>
    + MapElements<T>
    + ReduceElements<T>
    + AllocateZeros<T>
{
}

impl Device<f32> for Cpu {}
impl<T: CountElements, const M: usize> Device<[T; M]> for Cpu where Cpu: Device<T> {}
