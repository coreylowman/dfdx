use crate::prelude::*;

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
/// # use dfdx::*;
/// # let dev: Cpu = Default::default();
/// let m: MinPoolGlobal = Default::default();
/// let _: Tensor<Rank1<5>, f32, _> = m.forward(dev.zeros::<Rank3<5, 16, 8>>());
/// let _: Tensor<Rank2<10, 5>, f32, _> = m.forward(dev.zeros::<Rank4<10, 5, 16, 8>>());
/// ```
#[derive(Default, Debug, Clone, Copy, CustomModule)]
pub struct MinPoolGlobal;

impl<C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(C, H, W), E, D, T>> for MinPoolGlobal
{
    type Output = Tensor<(C,), E, D, T>;
    fn try_forward(
        &self,
        input: Tensor<(C, H, W), E, D, T>,
    ) -> Result<Self::Output, crate::tensor::Error> {
        input.try_min()
    }
}

impl<B: Dim, C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(B, C, H, W), E, D, T>> for MinPoolGlobal
{
    type Output = Tensor<(B, C), E, D, T>;
    fn try_forward(
        &self,
        input: Tensor<(B, C, H, W), E, D, T>,
    ) -> Result<Self::Output, crate::tensor::Error> {
        input.try_min()
    }
}
