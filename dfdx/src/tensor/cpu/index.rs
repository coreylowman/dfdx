use crate::{
    shapes::{Shape, Unit},
    tensor::{Cpu, Tensor},
};
use std::sync::Arc;

pub(crate) fn index_to_i<S: Shape>(shape: &S, strides: &S::Concrete, index: S::Concrete) -> usize {
    let sizes = shape.concrete();
    for (i, idx) in index.into_iter().enumerate() {
        if idx >= sizes[i] {
            panic!("Index out of bounds: index={index:?} shape={shape:?}");
        }
    }
    strides.into_iter().zip(index).map(|(a, b)| a * b).sum()
}

impl<S: Shape, E: Unit, T> std::ops::Index<S::Concrete> for Tensor<S, E, Cpu, T> {
    type Output = E;
    #[inline(always)]
    fn index(&self, index: S::Concrete) -> &Self::Output {
        let i = index_to_i(&self.shape, &self.strides, index);
        &self.data[i]
    }
}

impl<S: Shape, E: Unit, T> std::ops::IndexMut<S::Concrete> for Tensor<S, E, Cpu, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: S::Concrete) -> &mut Self::Output {
        let i = index_to_i(&self.shape, &self.strides, index);
        let data = Arc::make_mut(&mut self.data);
        &mut data[i]
    }
}
