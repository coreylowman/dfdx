use super::device::StridedArray;
use crate::arrays::{Dtype, Shape};
use std::sync::Arc;

impl<const N: usize, S: Shape<Concrete = [usize; N]>, E: Dtype> std::ops::Index<S::Concrete>
    for StridedArray<S, E>
{
    type Output = E;
    #[inline(always)]
    fn index(&self, index: S::Concrete) -> &Self::Output {
        let shape = self.shape.concrete();
        for (i, idx) in index.iter().enumerate() {
            if idx >= &shape[i] {
                panic!("Index {i} out of bounds: index={index:?} shape={shape:?}");
            }
        }
        let i: usize = self
            .strides
            .0
            .iter()
            .zip(index.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        &self.data[i]
    }
}

impl<const N: usize, S: Shape<Concrete = [usize; N]>, E: Dtype> std::ops::IndexMut<S::Concrete>
    for StridedArray<S, E>
{
    #[inline(always)]
    fn index_mut(&mut self, index: S::Concrete) -> &mut Self::Output {
        let shape = self.shape.concrete();
        for (i, idx) in index.iter().enumerate() {
            if idx >= &shape[i] {
                panic!("Index {i} out of bounds: index={index:?} shape={shape:?}");
            }
        }
        let i: usize = self
            .strides
            .0
            .iter()
            .zip(index.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let data = Arc::make_mut(&mut self.data);
        &mut data[i]
    }
}
