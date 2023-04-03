use crate::{
    shapes::*,
    tensor::{Cuda, Tensor},
};

use std::sync::Arc;

use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};

use super::{Bilinear, NearestNeighbor, UpscaleMethod};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/upscale2d.ptx"));

unsafe impl DeviceRepr for super::Upscale2DOp {}

fn make_4d<S: Shape>(strides: S::Concrete, pad: usize) -> [usize; 4] {
    match S::NUM_DIMS {
        3 => [pad, strides[0], strides[1], strides[2]],
        4 => [strides[0], strides[1], strides[2], strides[3]],
        _ => panic!("Only implemented for 3d & 4d arrays"),
    }
}

trait HasCudaKernel<E, Mode> {
    const FWD: &'static str;
    const BWD: &'static str;
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

        let inp_strides = self.dev.htod_copy(make_4d::<I>(inp.strides, 0).into())?;
        let inp_sizes = self
            .dev
            .htod_copy(make_4d::<I>(inp.shape.concrete(), 1).into())?;
        let out_strides = self.dev.htod_copy(make_4d::<O>(out.strides, 0).into())?;
        let out_sizes = self
            .dev
            .htod_copy(make_4d::<O>(out.shape.concrete(), 1).into())?;
        let fwd_fn = self.dev.get_func(Self::FWD, Self::FWD).unwrap();
        let cfg = LaunchConfig::for_num_elems(out.shape().num_elements() as u32);
        let params = (
            op,                           // const Pool2dOp op,
            &inp_strides,                 // const size_t *inp_strides,
            &inp_sizes,                   // const size_t *inp_sizes,
            &out_strides,                 // const size_t *out_strides,
            &out_sizes,                   // const size_t *out_sizes,
            inp.data.as_ref(),            // const float *inp,
            Arc::make_mut(&mut out.data), // float *out
        );
        unsafe { fwd_fn.launch(cfg, params) }?;
        Ok(())
    }
    fn backward<I: Shape, O: Shape>(
        &self,
        op: super::Upscale2DOp,
        inp: &Tensor<I, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let inp_strides = self.dev.htod_copy(make_4d::<I>(inp.strides, 0).into())?;
        let inp_sizes = self
            .dev
            .htod_copy(make_4d::<I>(inp.shape.concrete(), 1).into())?;
        let out_strides = self.dev.htod_copy(make_4d::<O>(out.strides, 0).into())?;
        let out_sizes = self
            .dev
            .htod_copy(make_4d::<O>(out.shape.concrete(), 1).into())?;
        let bwd_fn = self.dev.get_func(Self::FWD, Self::BWD).unwrap();
        let cfg = LaunchConfig::for_num_elems(out.shape().num_elements() as u32);
        let params = (
            op,                // const Pool2dOp op,
            &inp_strides,      // const size_t *inp_strides,
            &inp_sizes,        // const size_t *inp_sizes,
            &out_strides,      // const size_t *out_strides,
            &out_sizes,        // const size_t *out_sizes,
            inp.data.as_ref(), // const float *inp,
            grad_inp,          // float *grad_inp,
            out.data.as_ref(), // const float *out,
            grad_out,          // const float *grad_out
        );
        unsafe { bwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}
