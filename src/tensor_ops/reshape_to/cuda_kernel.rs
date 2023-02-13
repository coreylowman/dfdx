use crate::{
    shapes::*,
    tensor::cuda::{Cuda, CudaArray},
};
use cudarc::driver::{LaunchAsync, LaunchConfig};
use std::sync::Arc;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/reshape.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "reshape_f32";
    const FNS: &'static [&'static str] = &["reshape_fwd_f32", "reshape_bwd_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "reshape_f64";
    const FNS: &'static [&'static str] = &["reshape_fwd_f64", "reshape_bwd_f64"];
}

impl<E: Dtype> super::ReshapeKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<Src: Shape, Dst: Shape>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: HasSameNumelAs<Dst>,
    {
        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let numel = inp.data.len();
        let mut storage = unsafe { self.dev.alloc_async::<E>(numel) }?;

        let inp_dims = self.dev.take_async(inp.shape.concrete().into())?;
        let dst_dims = self.dev.take_async(dst.concrete().into())?;
        let inp_strides = self.dev.take_async(inp.strides.into())?;
        let dst_strides = self.dev.take_async(dst.strides().into())?;

        let fwd_fn = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,             // const size_t numel,
            inp.data.as_ref(), // const float *inp,
            Src::NUM_DIMS,     // const size_t inp_num_dims,
            &inp_dims,         // const size_t *inp_dims,
            &inp_strides,      // const size_t *inp_strides,
            &mut storage,      // float *out
            Dst::NUM_DIMS,     // const size_t out_num_dims,
            &dst_dims,         // const size_t *out_dims,
            &dst_strides,      // const size_t *out_strides,
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;

        Ok(CudaArray {
            data: Arc::new(storage),
            shape: dst,
            strides: dst.strides(),
        })
    }

    fn backward<Src: Shape, Dst: Shape>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: HasSameNumelAs<Dst>,
    {
        let bwd_fn = self.dev.get_func(Self::MOD, Self::FNS[1]).unwrap();
        let numel = grad_inp.data.len();

        let inp_dims = self.dev.take_async(grad_inp.shape.concrete().into())?;
        let out_dims = self.dev.take_async(grad_out.shape.concrete().into())?;
        let inp_strides = self.dev.take_async(grad_inp.strides.into())?;
        let out_strides = self.dev.take_async(grad_out.strides.into())?;

        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,                             // const size_t numel,
            Arc::make_mut(&mut grad_inp.data), // float *grad_inp,
            Src::NUM_DIMS,                     // const size_t inp_num_dims,
            &inp_dims,                         // const size_t *inp_dims,
            &inp_strides,                      // const size_t *inp_strides,
            grad_out.data.as_ref(),            // const float *grad_out,
            Dst::NUM_DIMS,                     // const size_t out_num_dims,
            &out_dims,                         // const size_t *out_dims,
            &out_strides,                      // const size_t *out_strides
        );
        unsafe { bwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
}
