use crate::{shapes::*, tensor::cuda::Cuda};

use std::sync::Arc;

use cudarc::driver::{AsKernelParam, LaunchAsync, LaunchConfig};

const MODULE_NAME: &str = "pool2d";
const AVG_FWD: &str = "avg_pool2d_forward";
const AVG_BWD: &str = "avg_pool2d_backward";
const ALL_FN_NAMES: [&str; 2] = [AVG_FWD, AVG_BWD];
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/pool2d.ptx"));

unsafe impl AsKernelParam for super::Pool2DOp {}

fn make_4d<S: Shape>(strides: S::Concrete) -> [usize; 4] {
    match S::NUM_DIMS {
        3 => [0, strides[0], strides[1], strides[2]],
        4 => [strides[0], strides[1], strides[2], strides[3]],
        _ => panic!("Only implemented for 3d & 4d arrays"),
    }
}

impl super::AvgPool2DKernel<f32> for Cuda {
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Self::Storage<I, f32>,
        out: &mut Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(MODULE_NAME, AVG_FWD) {
            self.dev
                .load_ptx(PTX_SRC.into(), MODULE_NAME, &ALL_FN_NAMES)?;
        }

        let inp_strides = self.dev.take_async(make_4d::<I>(inp.strides).into())?;
        let out_strides = self.dev.take_async(make_4d::<O>(out.strides).into())?;
        let fwd_fn = self.dev.get_func(MODULE_NAME, AVG_FWD).unwrap();
        let cfg = LaunchConfig::for_num_elems(out.shape().num_elements() as u32);
        let params = (
            op,                           // const Pool2dOp op,
            &inp_strides,                 // const size_t *inp_strides,
            &out_strides,                 // const size_t *out_strides,
            inp.data.as_ref(),            // const float *inp,
            Arc::make_mut(&mut out.data), // float *out
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
    fn backward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Self::Storage<I, f32>,
        grad_inp: &mut Self::Storage<I, f32>,
        out: &Self::Storage<O, f32>,
        grad_out: &Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        let inp_strides = self.dev.take_async(make_4d::<I>(inp.strides).into())?;
        let out_strides = self.dev.take_async(make_4d::<O>(out.strides).into())?;
        let bwd_fn = self.dev.get_func(MODULE_NAME, AVG_BWD).unwrap();
        let cfg = LaunchConfig::for_num_elems(grad_inp.shape().num_elements() as u32);
        let params = (
            op,                                // const Pool2dOp op,
            &inp_strides,                      // const size_t *inp_strides,
            &out_strides,                      // const size_t *out_strides,
            inp.data.as_ref(),                 // const float *inp,
            Arc::make_mut(&mut grad_inp.data), // float *grad_inp,
            out.data.as_ref(),                 // const float *out,
            grad_out.data.as_ref(),            // const float *grad_out
        );
        unsafe { bwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
}

impl super::MaxPool2DKernel<f32> for Cuda {
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Self::Storage<I, f32>,
        out: &mut Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        todo!();
    }
    fn backward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Self::Storage<I, f32>,
        grad_inp: &mut Self::Storage<I, f32>,
        out: &Self::Storage<O, f32>,
        grad_out: &Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        todo!();
    }
}

impl super::MinPool2DKernel<f32> for Cuda {
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Self::Storage<I, f32>,
        out: &mut Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        todo!();
    }
    fn backward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Self::Storage<I, f32>,
        grad_inp: &mut Self::Storage<I, f32>,
        out: &Self::Storage<O, f32>,
        grad_out: &Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        todo!();
    }
}
