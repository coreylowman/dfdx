use super::Cpu;
use super::*;
use crate::prelude::Device;

pub trait IsNdArray {
    type Array: 'static + Sized + Clone + CountElements;
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
