use crate::*;
use dfdx::{
    shapes::{Const, Dim, Dtype, HasShape},
    tensor::{Tape, Tensor},
    tensor_ops::{Device, GenericUpscale2D, NearestNeighbor, Upscale2DKernel, UpscaleMethod},
};

#[derive(Debug, Default, Clone, CustomModule)]
pub struct Upscale2D<
    OutHeight: Dim,
    OutWidth: Dim = OutHeight,
    Method: UpscaleMethod = NearestNeighbor,
> {
    pub out_width: OutWidth,
    pub out_height: OutHeight,
    pub method: Method,
}

pub type Upscale2DConst<const OH: usize, const OW: usize = OH, M = NearestNeighbor> =
    Upscale2D<Const<OH>, Const<OW>, M>;

impl<H: Dim, W: Dim, M: UpscaleMethod, Img: GenericUpscale2D<M>> Module<Img>
    for Upscale2D<H, W, M>
{
    type Output = Img::Output<H, W>;
    type Error = Img::Err;

    fn try_forward(&self, x: Img) -> Result<Self::Output, Img::Err> {
        x.generic_upscale2d_like(self.method, self.out_height, self.out_width)
    }
}

#[derive(Debug, Default, Clone, CustomModule)]
pub struct Upscale2DBy<H: Dim, W: Dim = H, Method: UpscaleMethod = NearestNeighbor> {
    pub height_factor: H,
    pub width_factor: W,
    pub method: Method,
}

pub type Upscale2DByConst<const HF: usize, const WF: usize = HF, M = NearestNeighbor> =
    Upscale2DBy<Const<HF>, Const<WF>, M>;

impl<HF: Dim, WF: Dim, C: Dim, H, W, M: UpscaleMethod, E: Dtype, D, T: Tape<E, D>>
    Module<Tensor<(C, H, W), E, D, T>> for Upscale2DBy<HF, WF, M>
where
    H: Dim + std::ops::Mul<HF>,
    H::Output: Dim,
    W: Dim + std::ops::Mul<WF>,
    W::Output: Dim,
    D: Device<E> + Upscale2DKernel<E, M>,
{
    type Output = Tensor<(C, H::Output, W::Output), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, x: Tensor<(C, H, W), E, D, T>) -> Result<Self::Output, Self::Error> {
        let (_c, h, w) = *x.shape();
        let h = h * self.height_factor;
        let w = w * self.width_factor;
        x.generic_upscale2d_like(self.method, h, w)
    }
}

impl<HF: Dim, WF: Dim, B: Dim, C: Dim, H, W, M: UpscaleMethod, E: Dtype, D, T>
    Module<Tensor<(B, C, H, W), E, D, T>> for Upscale2DBy<HF, WF, M>
where
    H: Dim + std::ops::Mul<HF>,
    H::Output: Dim,
    W: Dim + std::ops::Mul<WF>,
    W::Output: Dim,
    D: Device<E> + Upscale2DKernel<E, M>,
    T: 'static + Tape<E, D>,
{
    type Output = Tensor<(B, C, H::Output, W::Output), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, x: Tensor<(B, C, H, W), E, D, T>) -> Result<Self::Output, Self::Error> {
        let (_b, _c, h, w) = *x.shape();
        let h = h * self.height_factor;
        let w = w * self.width_factor;
        x.generic_upscale2d_like(self.method, h, w)
    }
}
