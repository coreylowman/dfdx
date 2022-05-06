use super::*;

pub trait IsNdArray {
    type ArrayType: 'static
        + Sized
        + Clone
        + ZipMapElements<Self::ArrayType>
        + MapElements
        + ZeroElements
        + CountElements
        + ReduceElements
        + FillElements;
}
