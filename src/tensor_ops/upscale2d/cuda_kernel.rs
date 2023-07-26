use crate::{
    dtypes::*,
    shapes::*,
    tensor::{launch_cfg, Cuda, Tensor},
};

use std::sync::Arc;

use cudarc::driver::{DeviceRepr, LaunchAsync};

use super::{Bilinear, NearestNeighbor, UpscaleMethod};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/upscale2d.ptx"));

unsafe impl DeviceRepr for super::Upscale2DOp {}

fn make_4d<S: Shape>(strides: S::Concrete) -> [usize; 4] {
    match S::NUM_DIMS {
        3 => [0, strides[0], strides[1], strides[2]],
        4 => [strides[0], strides[1], strides[2], strides[3]],
        _ => panic!("Only implemented for 3d & 4d arrays"),
    }
}

trait HasCudaKernel<E, Mode> {
    const FWD: &'static str;
    const BWD: &'static str;
}
#[cfg(feature = "f16")]
impl HasCudaKernel<f16, NearestNeighbor> for Cuda {
    const FWD: &'static str = "nearest_upscale2d_fwd_f16";
    const BWD: &'static str = "nearest_upscale2d_bwd_f16";
}
#[cfg(feature = "f16")]
impl HasCudaKernel<f16, Bilinear> for Cuda {
    const FWD: &'static str = "bilinear_upscale2d_fwd_f16";
    const BWD: &'static str = "bilinear_upscale2d_bwd_f16";
}
#[cfg(feature = "f16")]
impl HasCudaKernel<AMP<f16>, NearestNeighbor> for Cuda {
    const FWD: &'static str = "nearest_upscale2d_fwd_f16";
    const BWD: &'static str = "nearest_upscale2d_bwd_f16";
}
#[cfg(feature = "f16")]
impl HasCudaKernel<AMP<f16>, Bilinear> for Cuda {
    const FWD: &'static str = "bilinear_upscale2d_fwd_f16";
    const BWD: &'static str = "bilinear_upscale2d_bwd_f16";
}
impl HasCudaKernel<f32, NearestNeighbor> for Cuda {
    const FWD: &'static str = "nearest_upscale2d_fwd_f32";
    const BWD: &'static str = "nearest_upscale2d_bwd_f32";
}
impl HasCudaKernel<f32, Bilinear> for Cuda {
    const FWD: &'static str = "bilinear_upscale2d_fwd_f32";
    const BWD: &'static str = "bilinear_upscale2d_bwd_f32";
}
impl HasCudaKernel<f64, NearestNeighbor> for Cuda {
    const FWD: &'static str = "nearest_upscale2d_fwd_f64";
    const BWD: &'static str = "nearest_upscale2d_bwd_f64";
}
impl HasCudaKernel<f64, Bilinear> for Cuda {
    const FWD: &'static str = "bilinear_upscale2d_fwd_f64";
    const BWD: &'static str = "bilinear_upscale2d_bwd_f64";
}
impl<E: Dtype, Mode: UpscaleMethod> super::Upscale2DKernel<E, Mode> for Cuda
where
    Self: HasCudaKernel<E, Mode>,
{
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Upscale2DOp,
        inp: &Tensor<I, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::FWD, Self::FWD) {
            self.dev
                .load_ptx(PTX_SRC.into(), Self::FWD, &[Self::FWD, Self::BWD])?;
        }

        let strides = self.dev.htod_copy(make_4d::<I>(inp.strides).into())?;
        let fwd_fn = self.dev.get_func(Self::FWD, Self::FWD).unwrap();
        let cfg = launch_cfg::<128>(out.shape().num_elements() as u32);
        let params = (
            op,
            &strides,
            inp.data.as_ref(),
            Arc::make_mut(&mut out.data),
        );
        unsafe { fwd_fn.launch(cfg, params) }?;
        Ok(())
    }
    fn backward<I: Shape, O: Shape>(
        &self,
        op: super::Upscale2DOp,
        inp: &Tensor<I, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let strides = self.dev.htod_copy(make_4d::<I>(inp.strides).into())?;
        let bwd_fn = self.dev.get_func(Self::FWD, Self::BWD).unwrap();
        let cfg = launch_cfg::<128>(out.shape().num_elements() as u32);
        let params = (op, &strides, grad_inp, grad_out);
        unsafe { bwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}
