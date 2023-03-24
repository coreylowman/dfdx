use crate::prelude::{Tensor, Dtype, Rank3, Tape, ZerosTensor, Upscale2DKernel, HasErr};
#[cfg(feature = "nightly")]
use crate::prelude::{UpscaleMethod, ConstUpscale2D, NearestNeighbor};

#[allow(unused)]
use super::{BuildModule, Module, NonMutableModule, ZeroSizedModule};

#[derive(Debug, Default, Clone)]
pub struct Upscale2D<const OH: usize, const OW: usize = OH, M: UpscaleMethod = NearestNeighbor>(M);

impl<const OH: usize, const OW: usize, M: UpscaleMethod> ZeroSizedModule for Upscale2D<OH, OW, M> {}
impl<const OH: usize, const OW: usize, M: UpscaleMethod> NonMutableModule for Upscale2D<OH, OW, M> {}


#[cfg(feature = "nightly")]
impl<const OH: usize, const OW: usize, M: UpscaleMethod, Img: ConstUpscale2D<M>> Module<Img>
    for Upscale2D<OH, OW, M>
{
    type Output = Img::Output<OH, OW>;
    type Error = Img::Err;

    fn try_forward(&self, x: Img) -> Result<Self::Output, Img::Err> {
        x.try_upscale2d()
    }
}

#[derive(Debug, Default, Clone)]
pub struct Upscale2DBy<const H: usize, const W: usize = H, M: UpscaleMethod = NearestNeighbor>(M);

impl<const H: usize, const W: usize, M: UpscaleMethod> ZeroSizedModule for Upscale2DBy<H, W, M> {}
impl<const H: usize, const W: usize, M: UpscaleMethod> NonMutableModule for Upscale2DBy<H, W, M> {}


#[cfg(feature = "nightly")]
impl<const H: usize, const W: usize, const IH:usize, const IW: usize, const C: usize,
E: Dtype,
M: UpscaleMethod,
D: Upscale2DKernel<E, M> + ZerosTensor<E>,
T: 'static + Tape<E, D>,> Module<Tensor<Rank3<C, IH, IW>, E, D, T>>
    for Upscale2DBy<H, W, M>
where Tensor<Rank3<C, {IH*H}, {IW*W}>, E, D, T>: Sized
{
    type Output = Tensor<Rank3<C, {IH*H}, {IW*W}>, E, D, T>;
    type Error = <Self::Output as HasErr>::Err;

    fn try_forward(&self, x: Tensor<Rank3<C, IH, IW>, E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_upscale2d()
    }
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shapes::*, tensor::*, tests::*};

    #[test]
    fn test_upscale2d() {
        let dev: TestDevice = Default::default();
        let x: Tensor<Rank3<3, 4, 4>, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank3<3, 8, 8>, _, _> = Upscale2D::<8>::default().forward(x.clone());
        let _: Tensor<Rank3<3, 8, 12>, _, _> = Upscale2D::<8, 12>::default().forward(x.clone());
        let _: Tensor<Rank3<3, 9, 9>, _, _> = Upscale2D::<9, 9, NearestNeighbor>::default().forward(x.clone());
    }

    #[test]
    fn test_upscale2dby() {
        let dev: TestDevice = Default::default();
        let x: Tensor<Rank3<3, 4, 4>, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank3<3, 8, 8>, _, _> = Upscale2DBy::<2>::default().forward(x.clone());
        let _: Tensor<Rank3<3, 8, 12>, _, _> = Upscale2DBy::<2, 3>::default().forward(x.clone());
        let _: Tensor<Rank3<3, 12, 12>, _, _> = Upscale2DBy::<3, 3, NearestNeighbor>::default().forward(x.clone());
    }
}
