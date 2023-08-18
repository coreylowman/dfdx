use crate::*;
use dfdx::{
    shapes::{Const, Dim},
    tensor_ops::TryPool2D,
};

#[derive(Debug, Default, Clone, CustomModule)]
pub struct MaxPool2D<
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

pub type MaxPool2DConst<
    const KERNEL_SIZE: usize,
    const STRIDE: usize = 1,
    const PADDING: usize = 0,
    const DILATION: usize = 1,
> = MaxPool2D<Const<KERNEL_SIZE>, Const<STRIDE>, Const<PADDING>, Const<DILATION>>;

impl<K: Dim, S: Dim, P: Dim, L: Dim, Img: TryPool2D<K, S, P, L>> crate::Module<Img>
    for MaxPool2D<K, S, P, L>
{
    type Output = Img::Pooled;
    type Error = Img::Error;

    fn try_forward(&self, x: Img) -> Result<Self::Output, Self::Error> {
        x.try_pool2d(
            dfdx::tensor_ops::Pool2DKind::Max,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}
