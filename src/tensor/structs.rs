use super::tape_holders::NoTape;
use crate::unique_id::UniqueId;

/// A 0d [Tensor] with shape (). Backed by data `f32`.
#[derive(Debug, Clone)]
pub struct Tensor0D<TapeHolder = NoTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: Box<f32>,
    pub(crate) tape: TapeHolder,
}

/// A 1d [Tensor] with shape (M, ). Backed by data `[f32; M]`.
#[derive(Debug, Clone)]
pub struct Tensor1D<const N: usize, TapeHolder = NoTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: Box<[f32; N]>,
    pub(crate) tape: TapeHolder,
}

/// A 2d [Tensor] with shape (M, N). Backed by data `[[f32; N]; M]`.
#[derive(Debug, Clone)]
pub struct Tensor2D<const M: usize, const N: usize, TapeHolder = NoTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: Box<[[f32; N]; M]>,
    pub(crate) tape: TapeHolder,
}

/// A 3d [Tensor] with shape (M, N, O). Backed by data `[[[f32; O]; N]; M]`.
#[derive(Debug, Clone)]
pub struct Tensor3D<const M: usize, const N: usize, const O: usize, TapeHolder = NoTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: Box<[[[f32; O]; N]; M]>,
    pub(crate) tape: TapeHolder,
}

/// A 4d [Tensor] with shape (M, N, O, P). Backed by data `[[[[f32; P]; O]; N]; M]`.
#[derive(Debug, Clone)]
pub struct Tensor4D<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    TapeHolder = NoTape,
> {
    pub(crate) id: UniqueId,
    pub(crate) data: Box<[[[[f32; P]; O]; N]; M]>,
    pub(crate) tape: TapeHolder,
}
