use crate::{
    shapes::*,
    tensor::{launch_cfg, Cuda, Tensor},
};
use cudarc::driver::{CudaSlice, LaunchAsync};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/choose.ptx"));

pub(crate) trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "choose_f32";
    const FNS: &'static [&'static str] = &["choose_fwd_f32", "choose_bwd_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "choose_f64";
    const FNS: &'static [&'static str] = &["choose_fwd_f64", "choose_bwd_f64"];
}

impl<E: Dtype> super::ChooseKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<S: Shape>(
        &self,
        cond: &Tensor<S, bool, Self>,
        lhs: &Tensor<S, E, Self>,
        rhs: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let shape = lhs.shape;
        let strides = lhs.shape.strides();
        let numel = shape.num_elements();

        let mut storage = unsafe { self.dev.alloc::<E>(numel) }?;

        let dims: CudaSlice<usize> = self.dev.htod_copy(shape.concrete().into())?;
        let cond_strides: CudaSlice<usize> = self.dev.htod_copy(cond.strides.into())?;
        let lhs_strides: CudaSlice<usize> = self.dev.htod_copy(lhs.strides.into())?;
        let rhs_strides: CudaSlice<usize> = self.dev.htod_copy(rhs.strides.into())?;

        let fwd_fn = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();
        let cfg = launch_cfg(numel as u32);
        let params = (
            numel,              // const size_t numel,
            S::NUM_DIMS,        // const size_t num_dims,
            &dims,              // const size_t *dims,
            cond.data.as_ref(), // const bool *cond,
            &cond_strides,      // const size_t *cond_strides,
            lhs.data.as_ref(),  // const float *lhs,
            &lhs_strides,       // const size_t *lhs_strides,
            rhs.data.as_ref(),  // const float *rhs,
            &rhs_strides,       // const size_t *rhs_strides,
            &mut storage,       // float *out,
        );
        unsafe { fwd_fn.launch(cfg, params) }?;
        Ok(self.build_tensor(shape, strides, storage))
    }

    fn backward<S: Shape>(
        &self,
        cond: &Tensor<S, bool, Self>,
        lhs: &Tensor<S, E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<S, E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let bwd_fn = self.dev.get_func(Self::MOD, Self::FNS[1]).unwrap();
        let numel = cond.shape.num_elements();

        let dims: CudaSlice<usize> = self.dev.htod_copy(cond.shape.concrete().into())?;
        let lhs_strides: CudaSlice<usize> = self.dev.htod_copy(lhs.strides.into())?;
        let cond_strides: CudaSlice<usize> = self.dev.htod_copy(cond.strides.into())?;
        let rhs_strides: CudaSlice<usize> = self.dev.htod_copy(rhs.strides.into())?;

        let cfg = launch_cfg(numel as u32);
        let params = (
            numel,              // const size_t numel,
            S::NUM_DIMS,        // const size_t num_dims,
            &dims,              // const size_t *dims,
            cond.data.as_ref(), // const bool *cond,
            &cond_strides,      // const size_t *cond_strides,
            grad_lhs,           // float *grad_lhs,
            &lhs_strides,       // const size_t *lhs_strides,
            grad_rhs,           // float *grad_rhs,
            &rhs_strides,       // const size_t *rhs_strides,
            grad_out,           // const float *grad_out,
        );
        unsafe { bwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}
