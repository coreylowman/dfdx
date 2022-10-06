use super::{LoadFromNpz, Module, ResetParams, SaveToNpz};
use crate::gradients::*;
use crate::tensor::*;

/// Applies min pooling over an entire image, fully reducing the height and width
/// dimensions:
/// - Reduces 2d (C, L) to 1d (C, )
/// - Reduces 3d (C, H, W) to 1d (C, )
/// - Reduces 4d (B, C, H, W) to 2d (B, C)
///
/// **Pytorch equivalent**: `torch.nn.AdaptiveMinPool2d(1)` followed by a flatten.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let m: MinPoolGlobal = Default::default();
/// let _: Tensor1D<5> = m.forward(Tensor3D::<5, 16, 8>::zeros());
/// let _: Tensor2D<10, 5> = m.forward(Tensor4D::<10, 5, 16, 8>::zeros());
/// ```
#[derive(Clone, Copy, Default)]
pub struct MinPoolGlobal;

impl ResetParams for MinPoolGlobal {
    fn reset_params<R: rand::Rng>(&mut self, _: &mut R) {}
}
impl CanUpdateWithGradients for MinPoolGlobal {
    fn update<G: GradientProvider>(&mut self, _: &mut G, _: &mut UnusedTensors) {}
}
impl SaveToNpz for MinPoolGlobal {}
impl LoadFromNpz for MinPoolGlobal {}

impl<const C: usize, const L: usize, T: Tape> Module<Tensor2D<C, L, T>> for MinPoolGlobal {
    type Output = Tensor1D<C, T>;
    fn forward(&self, input: Tensor2D<C, L, T>) -> Self::Output {
        input.min()
    }
}

impl<const C: usize, const H: usize, const W: usize, T: Tape> Module<Tensor3D<C, H, W, T>>
    for MinPoolGlobal
{
    type Output = Tensor1D<C, T>;
    fn forward(&self, input: Tensor3D<C, H, W, T>) -> Self::Output {
        input.min()
    }
}

impl<const B: usize, const C: usize, const H: usize, const W: usize, T: Tape>
    Module<Tensor4D<B, C, H, W, T>> for MinPoolGlobal
{
    type Output = Tensor2D<B, C, T>;
    fn forward(&self, input: Tensor4D<B, C, H, W, T>) -> Self::Output {
        input.min()
    }
}
