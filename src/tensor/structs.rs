use super::tape_holders::NoTape;
use crate::gradients::UniqueId;

#[derive(Debug)]
pub struct Tensor0D<TapeHolder = NoTape> {
    pub(super) id: UniqueId,
    pub(super) data: f32,
    pub(super) tape: TapeHolder,
}

#[derive(Debug)]
pub struct Tensor1D<const N: usize, TapeHolder = NoTape> {
    pub(super) id: UniqueId,
    pub(super) data: [f32; N],
    pub(super) tape: TapeHolder,
}

#[derive(Debug)]
pub struct Tensor2D<const M: usize, const N: usize, TapeHolder = NoTape> {
    pub(super) id: UniqueId,
    pub(super) data: [[f32; N]; M],
    pub(super) tape: TapeHolder,
}

#[derive(Debug)]
pub struct Tensor3D<const M: usize, const N: usize, const O: usize, TapeHolder = NoTape> {
    pub(super) id: UniqueId,
    pub(super) data: [[[f32; O]; N]; M],
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
    pub(super) id: UniqueId,
    pub(super) data: [[[[f32; P]; O]; N]; M],
    pub(super) tape: TapeHolder,
}
