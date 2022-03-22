use super::tape_holders::NoTape;
use ndarray::{Array, Ix0, Ix1, Ix2, Ix3, Ix4};

#[derive(Debug)]
pub struct Tensor0D<TapeHolder = NoTape> {
    pub(super) id: usize,
    pub(super) data: Array<f32, Ix0>,
    pub(super) tape: TapeHolder,
}

#[derive(Debug)]
pub struct Tensor1D<const N: usize, TapeHolder = NoTape> {
    pub(super) id: usize,
    pub(super) data: Array<f32, Ix1>,
    pub(super) tape: TapeHolder,
}

#[derive(Debug)]
pub struct Tensor2D<const M: usize, const N: usize, TapeHolder = NoTape> {
    pub(super) id: usize,
    pub(super) data: Array<f32, Ix2>,
    pub(super) tape: TapeHolder,
}

#[derive(Debug)]
pub struct Tensor3D<const M: usize, const N: usize, const O: usize, TapeHolder = NoTape> {
    pub(super) id: usize,
    pub(super) data: Array<f32, Ix3>,
    pub(super) tape: TapeHolder,
}

#[derive(Debug)]
pub struct Tensor4D<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    TapeHolder = NoTape,
> {
    pub(super) id: usize,
    pub(super) data: Array<f32, Ix4>,
    pub(super) tape: TapeHolder,
}
