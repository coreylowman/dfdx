use super::structs::*;
use super::traits::*;
use ndarray::Array;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

fn unique_id() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

impl Tensor for Tensor0D {
    fn new(data: Array<f32, Self::Dimension>) -> Self {
        Self {
            id: unique_id(),
            data,
            tape: RefCell::new(None),
        }
    }
}

impl<const N: usize> Tensor for Tensor1D<N> {
    fn new(data: Array<f32, Self::Dimension>) -> Self {
        Self {
            id: unique_id(),
            data,
            tape: RefCell::new(None),
        }
    }
}

impl<const M: usize, const N: usize> Tensor for Tensor2D<M, N> {
    fn new(data: Array<f32, Self::Dimension>) -> Self {
        Self {
            id: unique_id(),
            data,
            tape: RefCell::new(None),
        }
    }
}

impl<const M: usize, const N: usize, const O: usize> Tensor for Tensor3D<M, N, O> {
    fn new(data: Array<f32, Self::Dimension>) -> Self {
        Self {
            id: unique_id(),
            data,
            tape: RefCell::new(None),
        }
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize> Tensor
    for Tensor4D<M, N, O, P>
{
    fn new(data: Array<f32, Self::Dimension>) -> Self {
        Self {
            id: unique_id(),
            data,
            tape: RefCell::new(None),
        }
    }
}

fn tensor_default<T: Tensor>() -> T {
    T::new(Array::zeros(T::SHAPE))
}

impl Default for Tensor0D {
    fn default() -> Self {
        tensor_default()
    }
}

impl<const N: usize> Default for Tensor1D<N> {
    fn default() -> Self {
        tensor_default()
    }
}

impl<const M: usize, const N: usize> Default for Tensor2D<M, N> {
    fn default() -> Self {
        tensor_default()
    }
}

impl<const M: usize, const N: usize, const O: usize> Default for Tensor3D<M, N, O> {
    fn default() -> Self {
        tensor_default()
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize> Default
    for Tensor4D<M, N, O, P>
{
    fn default() -> Self {
        tensor_default()
    }
}
