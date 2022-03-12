use crate::gradients::{GradientRef, GradientTape};
use ndarray::{Array, Ix0, Ix1, Ix2, Ix3, Ix4};
use std::cell::RefCell;

#[derive(Debug)]
pub struct GradientData {
    pub(super) grad_ref: Option<GradientRef>,
    pub(super) tape: Option<Box<GradientTape>>,
}

#[derive(Debug)]
pub struct Tensor0D {
    pub(super) data: Array<f32, Ix0>,
    pub(super) grad: RefCell<GradientData>,
}

#[derive(Debug)]
pub struct Tensor1D<const N: usize> {
    pub(super) data: Array<f32, Ix1>,
    pub(super) grad: RefCell<GradientData>,
}

#[derive(Debug)]
pub struct Tensor2D<const M: usize, const N: usize> {
    pub(super) data: Array<f32, Ix2>,
    pub(super) grad: RefCell<GradientData>,
}

#[derive(Debug)]
pub struct Tensor3D<const M: usize, const N: usize, const O: usize> {
    pub(super) data: Array<f32, Ix3>,
    pub(super) grad: RefCell<GradientData>,
}

#[derive(Debug)]
pub struct Tensor4D<const M: usize, const N: usize, const O: usize, const P: usize> {
    pub(super) data: Array<f32, Ix4>,
    pub(super) grad: RefCell<GradientData>,
}
