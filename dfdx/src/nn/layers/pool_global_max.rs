use crate::prelude::*;

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
/// # use dfdx_nn::*;
/// # let dev: Cpu = Default::default();
/// let m: MaxPoolGlobal = Default::default();
/// let _: Tensor<Rank1<5>, f32, _> = m.forward(dev.zeros::<Rank3<5, 16, 8>>());
/// let _: Tensor<Rank2<10, 5>, f32, _> = m.forward(dev.zeros::<Rank4<10, 5, 16, 8>>());
/// ```
#[derive(Default, Debug, Clone, Copy, CustomModule)]
pub struct MaxPoolGlobal;

impl<C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(C, H, W), E, D, T>> for MaxPoolGlobal
{
    type Output = Tensor<(C,), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(C, H, W), E, D, T>) -> Result<Self::Output, D::Err> {
        input.try_max()
    }
}

impl<B: Dim, C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(B, C, H, W), E, D, T>> for MaxPoolGlobal
{
    type Output = Tensor<(B, C), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(B, C, H, W), E, D, T>) -> Result<Self::Output, D::Err> {
        input.try_max()
    }
}
