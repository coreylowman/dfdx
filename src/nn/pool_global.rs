use crate::{arrays::*, gradients::*, optim::*, tensor::*, tensor_ops::*};

use super::{BuildModule, Module, ModuleMut};

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
#[derive(Clone, Copy, Default)]
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
#[derive(Clone, Copy, Default)]
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
#[derive(Clone, Copy, Default)]
pub struct MinPoolGlobal;

macro_rules! impl_pools {
    ($PoolTy:ty, $Method:ident) => {
        impl<D: Device<E>, E: Dtype> BuildModule<D, E> for $PoolTy {
            fn try_build(_: &D) -> Result<Self, <D>::Err> {
                Ok(Self)
            }
            fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
                Ok(())
            }
        }

        impl<D: Device<E>, E: Dtype> CanUpdateWithGradients<D, E> for $PoolTy {
            fn update<U>(&mut self, _: &mut U, _: &mut UnusedTensors) -> Result<(), <D>::Err>
            where
                U: UpdateParams<D, E>,
            {
                Ok(())
            }
        }

        impl<C: Dim, H: Dim, W: Dim, D: Device<f32>, T: Tape<D>>
            Module<Tensor<(C, H, W), f32, D, T>> for $PoolTy
        {
            type Output = Tensor<(C,), f32, D, T>;
            fn forward(&self, input: Tensor<(C, H, W), f32, D, T>) -> Self::Output {
                input.min()
            }
        }

        impl<B: Dim, C: Dim, H: Dim, W: Dim, D: Device<f32>, T: Tape<D>>
            Module<Tensor<(B, C, H, W), f32, D, T>> for $PoolTy
        {
            type Output = Tensor<(B, C), f32, D, T>;
            fn forward(&self, input: Tensor<(B, C, H, W), f32, D, T>) -> Self::Output {
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
