use crate::{gradients::*, shapes::*, tensor::*, tensor_ops::*};

use super::{BuildModule, Module, NonMutableModule, ZeroSizedModule};

/// Applies average pooling over an entire image, fully reducing the height and width
/// dimensions:
/// - Reduces 3d (C, H, W) to 1d (C, )
/// - Reduces 4d (B, C, H, W) to 2d (B, C)
///
/// **Pytorch equivalent**: `torch.nn.AdaptiveAvgPool2d(1)` followed by a flatten.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let m: AvgPoolGlobal = Default::default();
/// let _: Tensor<Rank1<5>, f32, _> = m.forward(dev.zeros::<Rank3<5, 16, 8>>());
/// let _: Tensor<Rank2<10, 5>, f32, _> = m.forward(dev.zeros::<Rank4<10, 5, 16, 8>>());
/// ```
#[derive(Clone, Copy, Default)]
pub struct AvgPoolGlobal;

/// Applies max pooling over an entire image, fully reducing the height and width
/// dimensions:
/// - Reduces 3d (C, H, W) to 1d (C, )
/// - Reduces 4d (B, C, H, W) to 2d (B, C)
///
/// **Pytorch equivalent**: `torch.nn.AdaptiveMaxPool2d(1)` followed by a flatten.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let m: MaxPoolGlobal = Default::default();
/// let _: Tensor<Rank1<5>, f32, _> = m.forward(dev.zeros::<Rank3<5, 16, 8>>());
/// let _: Tensor<Rank2<10, 5>, f32, _> = m.forward(dev.zeros::<Rank4<10, 5, 16, 8>>());
/// ```
#[derive(Clone, Copy, Default)]
pub struct MaxPoolGlobal;

/// Applies min pooling over an entire image, fully reducing the height and width
/// dimensions:
/// - Reduces 3d (C, H, W) to 1d (C, )
/// - Reduces 4d (B, C, H, W) to 2d (B, C)
///
/// **Pytorch equivalent**: `torch.nn.AdaptiveMinPool2d(1)` followed by a flatten.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let m: MinPoolGlobal = Default::default();
/// let _: Tensor<Rank1<5>, f32, _> = m.forward(dev.zeros::<Rank3<5, 16, 8>>());
/// let _: Tensor<Rank2<10, 5>, f32, _> = m.forward(dev.zeros::<Rank4<10, 5, 16, 8>>());
/// ```
#[derive(Clone, Copy, Default)]
pub struct MinPoolGlobal;

macro_rules! impl_pools {
    ($PoolTy:ty, $Method:ident) => {
        impl ZeroSizedModule for $PoolTy {}
        impl NonMutableModule for $PoolTy {}

        impl<D: Device<E>, E: Dtype> BuildModule<D, E> for $PoolTy {
            fn try_build(_: &D) -> Result<Self, <D>::Err> {
                Ok(Default::default())
            }
        }

        impl<C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<D>>
            Module<Tensor<(C, H, W), E, D, T>> for $PoolTy
        {
            type Output = Tensor<(C,), E, D, T>;
            fn forward(&self, input: Tensor<(C, H, W), E, D, T>) -> Self::Output {
                input.min()
            }
        }

        impl<B: Dim, C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<D>>
            Module<Tensor<(B, C, H, W), E, D, T>> for $PoolTy
        {
            type Output = Tensor<(B, C), E, D, T>;
            fn forward(&self, input: Tensor<(B, C, H, W), E, D, T>) -> Self::Output {
                input.$Method()
            }
        }
    };
}

impl_pools!(AvgPoolGlobal, mean);
impl_pools!(MaxPoolGlobal, max);
impl_pools!(MinPoolGlobal, min);
