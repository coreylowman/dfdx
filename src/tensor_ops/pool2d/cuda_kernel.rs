use crate::{
    shapes::*,
    tensor::{launch_cfg, Cuda, Tensor},
};

use std::sync::Arc;

use cudarc::driver::{DeviceRepr, LaunchAsync};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/pool2d.ptx"));

unsafe impl DeviceRepr for super::Pool2DOp {}

fn make_4d<S: Shape>(strides: S::Concrete) -> [usize; 4] {
    match S::NUM_DIMS {
        3 => [0, strides[0], strides[1], strides[2]],
        4 => [strides[0], strides[1], strides[2], strides[3]],
        _ => panic!("Only implemented for 3d & 4d arrays"),
    }
}

macro_rules! pool_impl {
    ($Trait:tt<$TypeName:ty>, $Fwd:tt, $Bwd:tt) => {
        impl super::$Trait<$TypeName> for Cuda {
            fn forward<I: Shape, O: Shape>(
                &self,
                op: super::Pool2DOp,
                inp: &Tensor<I, $TypeName, Self>,
                out: &mut Tensor<O, $TypeName, Self>,
            ) -> Result<(), Self::Err> {
                if !self.dev.has_func($Fwd, $Fwd) {
                    self.dev.load_ptx(PTX_SRC.into(), $Fwd, &[$Fwd, $Bwd])?;
                }

                let inp_strides = self.dev.htod_copy(make_4d::<I>(inp.strides).into())?;
                let out_strides = self.dev.htod_copy(make_4d::<O>(out.strides).into())?;
                let fwd_fn = self.dev.get_func($Fwd, $Fwd).unwrap();
                let cfg = launch_cfg(out.shape().num_elements() as u32);
                let params = (
                    op,                           // const Pool2dOp op,
                    &inp_strides,                 // const size_t *inp_strides,
                    &out_strides,                 // const size_t *out_strides,
                    inp.data.as_ref(),            // const float *inp,
                    Arc::make_mut(&mut out.data), // float *out
                );
                unsafe { fwd_fn.launch(cfg, params) }?;
                Ok(())
            }
            fn backward<I: Shape, O: Shape>(
                &self,
                op: super::Pool2DOp,
                inp: &Tensor<I, $TypeName, Self>,
                grad_inp: &mut Self::Vec<$TypeName>,
                out: &Tensor<O, $TypeName, Self>,
                grad_out: &Self::Vec<$TypeName>,
            ) -> Result<(), Self::Err> {
                let inp_strides = self.dev.htod_copy(make_4d::<I>(inp.strides).into())?;
                let out_strides = self.dev.htod_copy(make_4d::<O>(out.strides).into())?;
                let bwd_fn = self.dev.get_func($Fwd, $Bwd).unwrap();
                let cfg = launch_cfg(inp.shape().num_elements() as u32);
                let params = (
                    op,                // const Pool2dOp op,
                    &inp_strides,      // const size_t *inp_strides,
                    &out_strides,      // const size_t *out_strides,
                    inp.data.as_ref(), // const float *inp,
                    grad_inp,          // float *grad_inp,
                    out.data.as_ref(), // const float *out,
                    grad_out,          // const float *grad_out
                );
                unsafe { bwd_fn.launch(cfg, params) }?;
                Ok(())
            }
        }
    };
}

pool_impl!(
    AvgPool2DKernel<f32>,
    "avg_pool2d_fwd_f32",
    "avg_pool2d_bwd_f32"
);
pool_impl!(
    MaxPool2DKernel<f32>,
    "max_pool2d_fwd_f32",
    "max_pool2d_bwd_f32"
);
pool_impl!(
    MinPool2DKernel<f32>,
    "min_pool2d_fwd_f32",
    "min_pool2d_bwd_f32"
);

pool_impl!(
    AvgPool2DKernel<f64>,
    "avg_pool2d_fwd_f64",
    "avg_pool2d_bwd_f64"
);
pool_impl!(
    MaxPool2DKernel<f64>,
    "max_pool2d_fwd_f64",
    "max_pool2d_bwd_f64"
);
pool_impl!(
    MinPool2DKernel<f64>,
    "min_pool2d_fwd_f64",
    "min_pool2d_bwd_f64"
);
