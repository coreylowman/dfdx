use crate::*;
use dfdx::{
    shapes::{Const, Dim},
    tensor_ops::TryPool2D,
};

/// Minimum pool with 2d kernel that operates on images (3d) and batches of images (4d).
/// Each patch reduces to the minimum of the values in the patch.
///
/// Generics:
/// - `KERNEL_SIZE`: The size of the kernel applied to both width and height of the images.
/// - `STRIDE`: How far to move the kernel each step. Defaults to `1`
/// - `PADDING`: How much zero padding to add around the images. Defaults to `0`.
/// - `DILATION` How dilated the kernel should be. Defaults to `1`.
#[derive(Debug, Default, Clone, CustomModule)]
pub struct MinPool2D<
    KernelSize: Dim,
    Stride: Dim = Const<1>,
    Padding: Dim = Const<0>,
    Dilation: Dim = Const<1>,
> {
    pub kernel_size: KernelSize,
    pub stride: Stride,
    pub padding: Padding,
    pub dilation: Dilation,
}

pub type MinPool2DConst<
    const KERNEL_SIZE: usize,
    const STRIDE: usize = 1,
    const PADDING: usize = 0,
    const DILATION: usize = 1,
> = MinPool2D<Const<KERNEL_SIZE>, Const<STRIDE>, Const<PADDING>, Const<DILATION>>;

impl<K: Dim, S: Dim, P: Dim, L: Dim, Img: TryPool2D<K, S, P, L>> crate::Module<Img>
    for MinPool2D<K, S, P, L>
{
    type Output = Img::Pooled;
    type Error = Img::Error;

    fn try_forward(&self, x: Img) -> Result<Self::Output, Self::Error> {
        x.try_pool2d(
            dfdx::tensor_ops::Pool2DKind::Min,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}
