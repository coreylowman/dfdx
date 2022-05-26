use crate::{gradients::NoTape, unique_id::UniqueId};

/// A 0d [Tensor] with shape (). Backed by data `f32`.
#[derive(Debug, Clone)]
pub struct Tensor0D<Tape = NoTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: Box<f32>,
    pub(crate) tape: Tape,
}

/// A 1d [Tensor] with shape (M, ). Backed by data `[f32; M]`.
#[derive(Debug, Clone)]
pub struct Tensor1D<const N: usize, Tape = NoTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: Box<[f32; N]>,
    pub(crate) tape: Tape,
}

/// A 2d [Tensor] with shape (M, N). Backed by data `[[f32; N]; M]`.
#[derive(Debug, Clone)]
pub struct Tensor2D<const M: usize, const N: usize, Tape = NoTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: Box<[[f32; N]; M]>,
    pub(crate) tape: Tape,
}

/// A 3d [Tensor] with shape (M, N, O). Backed by data `[[[f32; O]; N]; M]`.
#[derive(Debug, Clone)]
pub struct Tensor3D<const M: usize, const N: usize, const O: usize, Tape = NoTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: Box<[[[f32; O]; N]; M]>,
    pub(crate) tape: Tape,
}

/// A 4d [Tensor] with shape (M, N, O, P). Backed by data `[[[[f32; P]; O]; N]; M]`.
#[derive(Debug, Clone)]
pub struct Tensor4D<const M: usize, const N: usize, const O: usize, const P: usize, Tape = NoTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: Box<[[[[f32; P]; O]; N]; M]>,
    pub(crate) tape: Tape,
}
