use dfdx_macros::CanUpdateWithGradients;
use super::{Module, ModuleMut, ResetParams};
use crate::gradients::*;
use crate::tensor::*;

/// Applies average pooling over an entire image, fully reducing the height and width
/// dimensions:
/// - Reduces 2d (C, L) to 1d (C, )
/// - Reduces 3d (C, H, W) to 1d (C, )
/// - Reduces 4d (B, C, H, W) to 2d (B, C)
///
/// **Pytorch equivalent**: `torch.nn.AdaptiveAvgPool2d(1)` followed by a flatten.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let m: AvgPoolGlobal = Default::default();
/// let _: Tensor1D<5> = m.forward(Tensor3D::<5, 16, 8>::zeros());
/// let _: Tensor2D<10, 5> = m.forward(Tensor4D::<10, 5, 16, 8>::zeros());
/// ```
#[derive(Clone, Copy, Default, CanUpdateWithGradients)]
pub struct AvgPoolGlobal;

/// Applies max pooling over an entire image, fully reducing the height and width
/// dimensions:
/// - Reduces 2d (C, L) to 1d (C, )
/// - Reduces 3d (C, H, W) to 1d (C, )
/// - Reduces 4d (B, C, H, W) to 2d (B, C)
///
/// **Pytorch equivalent**: `torch.nn.AdaptiveMaxPool2d(1)` followed by a flatten.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let m: MaxPoolGlobal = Default::default();
/// let _: Tensor1D<5> = m.forward(Tensor3D::<5, 16, 8>::zeros());
/// let _: Tensor2D<10, 5> = m.forward(Tensor4D::<10, 5, 16, 8>::zeros());
/// ```
#[derive(Clone, Copy, Default, CanUpdateWithGradients)]
pub struct MaxPoolGlobal;

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
#[derive(Clone, Copy, Default, CanUpdateWithGradients)]
pub struct MinPoolGlobal;

macro_rules! impl_pools {
    ($PoolTy:ty, $Method:ident) => {
        impl ResetParams for $PoolTy {
            fn reset_params<R: rand::Rng>(&mut self, _: &mut R) {}
        }

        impl<const C: usize, const L: usize, T: Tape> Module<Tensor2D<C, L, T>> for $PoolTy {
            type Output = Tensor1D<C, T>;
            fn forward(&self, input: Tensor2D<C, L, T>) -> Self::Output {
                input.$Method()
            }
        }

        impl<const C: usize, const H: usize, const W: usize, T: Tape> Module<Tensor3D<C, H, W, T>>
            for $PoolTy
        {
            type Output = Tensor1D<C, T>;
            fn forward(&self, input: Tensor3D<C, H, W, T>) -> Self::Output {
                input.$Method()
            }
        }

        impl<const B: usize, const C: usize, const H: usize, const W: usize, T: Tape>
            Module<Tensor4D<B, C, H, W, T>> for $PoolTy
        {
            type Output = Tensor2D<B, C, T>;
            fn forward(&self, input: Tensor4D<B, C, H, W, T>) -> Self::Output {
                input.$Method()
            }
        }

        impl<T> ModuleMut<T> for $PoolTy
        where
            Self: Module<T>,
        {
            type Output = <Self as Module<T>>::Output;
            fn forward_mut(&mut self, input: T) -> Self::Output {
                self.forward(input)
            }
        }
    };
}

impl_pools!(AvgPoolGlobal, mean);
impl_pools!(MaxPoolGlobal, max);
impl_pools!(MinPoolGlobal, min);
