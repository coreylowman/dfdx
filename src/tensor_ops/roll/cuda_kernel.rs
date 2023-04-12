use crate::{
    shapes::{Dtype, Shape},
    tensor::*,
};

use cudarc::driver::{DeviceRepr, LaunchAsync};

unsafe impl DeviceRepr for super::RollOp {}

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/roll.ptx"));

trait HasCudaKernel<E> {
    const FNS: &'static [&'static str];
}
impl HasCudaKernel<f32> for Cuda {
    const FNS: &'static [&'static str] = &["roll_fwd_f32", "roll_bwd_f32"];
}
impl HasCudaKernel<f64> for Cuda {
    const FNS: &'static [&'static str] = &["roll_fwd_f64", "roll_bwd_f64"];
}

impl<E: Dtype> super::RollKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<S: Shape>(
        &self,
        op: super::RollOp,
        inp: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        if !self.dev.has_func(Self::FNS[0], Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::FNS[0], Self::FNS)?;
        }

        let numel = inp.shape.num_elements();
        let strides = inp.shape.strides();

        let mut out = unsafe { self.alloc_empty::<E>(numel) }?;
        let dims = self.dev.htod_copy(inp.shape.concrete().into())?;
        let inp_strides = self.dev.htod_copy(inp.strides.into())?;
        let out_strides = self.dev.htod_copy(strides.into())?;

        let fwd = self.dev.get_func(Self::FNS[0], Self::FNS[0]).unwrap();
        let cfg = launch_cfg::<128>(inp.shape.num_elements() as u32);
        let params = (
            op,
            S::NUM_DIMS,
            numel,
            &dims,
            &inp_strides,
            &out_strides,
            inp.data.as_ref(),
            &mut out,
        );
        unsafe { fwd.launch(cfg, params) }?;
        Ok(self.build_tensor(inp.shape, strides, out))
    }
    fn backward<S: Shape>(
        &self,
        op: super::RollOp,
        inp: &Tensor<S, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let numel = inp.shape.num_elements();
        let strides = inp.shape.strides();

        let dims = self.dev.htod_copy(inp.shape.concrete().into())?;
        let inp_strides = self.dev.htod_copy(inp.strides.into())?;
        let out_strides = self.dev.htod_copy(strides.into())?;

        let bwd = self.dev.get_func(Self::FNS[0], Self::FNS[1]).unwrap();
        let cfg = launch_cfg::<128>(inp.shape.num_elements() as u32);
        let params = (
            op,
            S::NUM_DIMS,
            numel,
            &dims,
            &inp_strides,
            &out_strides,
            grad_inp,
            grad_out,
        );
        unsafe { bwd.launch(cfg, params) }?;
        Ok(())
    }
}
