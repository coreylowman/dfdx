//! We use [std::rc::Rc] instead of [Box] here to reduce allocations when tensors are duplicated/cloned.
//!
//! See [#62](https://github.com/coreylowman/dfdx/issues/62) for more discussion.

use crate::{gradients::NoneTape, unique_id::UniqueId};

/// A 0d [super::Tensor] with shape (). Backed by data `f32`.
#[derive(Debug)]
pub struct Tensor0D<Tape = NoneTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: std::sync::Arc<f32>,
    pub(crate) tape: Tape,
}

/// A 1d [super::Tensor] with shape (M, ). Backed by data `[f32; M]`.
#[derive(Debug)]
pub struct Tensor1D<const N: usize, Tape = NoneTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: std::sync::Arc<[f32; N]>,
    pub(crate) tape: Tape,
}

/// A 2d [super::Tensor] with shape (M, N). Backed by data `[[f32; N]; M]`.
#[derive(Debug)]
pub struct Tensor2D<const M: usize, const N: usize, Tape = NoneTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: std::sync::Arc<[[f32; N]; M]>,
    pub(crate) tape: Tape,
}

/// A 3d [super::Tensor] with shape (M, N, O). Backed by data `[[[f32; O]; N]; M]`.
#[derive(Debug)]
pub struct Tensor3D<const M: usize, const N: usize, const O: usize, Tape = NoneTape> {
    pub(crate) id: UniqueId,
    pub(crate) data: std::sync::Arc<[[[f32; O]; N]; M]>,
    pub(crate) tape: Tape,
}

/// A 4d [super::Tensor] with shape (M, N, O, P). Backed by data `[[[[f32; P]; O]; N]; M]`.
#[derive(Debug)]
pub struct Tensor4D<const M: usize, const N: usize, const O: usize, const P: usize, Tape = NoneTape>
{
    pub(crate) id: UniqueId,
    pub(crate) data: std::sync::Arc<[[[[f32; P]; O]; N]; M]>,
    pub(crate) tape: Tape,
}
