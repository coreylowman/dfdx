use crate::{
    dtypes::*,
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

trait HasCudaKernel<E> {
    const FWD: &'static str;
    const BWD: &'static str;
}

#[cfg(feature = "f16")]
impl HasCudaKernel<f16> for Cuda {
    const FWD: &'static str = "pool2d_fwd_f16";
    const BWD: &'static str = "pool2d_bwd_f16";
}

#[cfg(feature = "f16")]
impl HasCudaKernel<AMP<f16>> for Cuda {
    const FWD: &'static str = "pool2d_fwd_f16";
    const BWD: &'static str = "pool2d_bwd_f16";
}

impl HasCudaKernel<f32> for Cuda {
    const FWD: &'static str = "pool2d_fwd_f32";
    const BWD: &'static str = "pool2d_bwd_f32";
}

impl HasCudaKernel<f64> for Cuda {
    const FWD: &'static str = "pool2d_fwd_f64";
    const BWD: &'static str = "pool2d_bwd_f64";
}

impl<E: Dtype> super::Pool2DKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn alloc<S: Shape>(&self, s: S) -> Result<Tensor<S, E, Self>, Self::Err> {
        let data = unsafe { self.alloc_empty::<E>(s.num_elements()) }?;
        Ok(self.build_tensor(s, s.strides(), data))
    }
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Tensor<I, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::FWD, Self::FWD) {
            self.dev
                .load_ptx(PTX_SRC.into(), Self::FWD, &[Self::FWD, Self::BWD])?;
        }

        let inp_strides = self.dev.htod_copy(make_4d::<I>(inp.strides).into())?;
        let out_strides = self.dev.htod_copy(make_4d::<O>(out.strides).into())?;
        let fwd_fn = self.dev.get_func(Self::FWD, Self::FWD).unwrap();
        let cfg = launch_cfg::<128>(out.shape().num_elements() as u32);
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
        inp: &Tensor<I, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let inp_strides = self.dev.htod_copy(make_4d::<I>(inp.strides).into())?;
        let out_strides = self.dev.htod_copy(make_4d::<O>(out.strides).into())?;
        let bwd_fn = self.dev.get_func(Self::FWD, Self::BWD).unwrap();
        let cfg = launch_cfg::<128>(inp.shape().num_elements() as u32);
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
