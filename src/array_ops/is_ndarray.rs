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

pub trait Array: std::ops::IndexMut<usize, Output = Self::Element> {
    const SIZE: usize;
    type Element;
}

impl<const M: usize> Array for [f32; M] {
    const SIZE: usize = M;
    type Element = f32;
}

impl<T: Array, const M: usize> Array for [T; M] {
    const SIZE: usize = M;
    type Element = T;
}
