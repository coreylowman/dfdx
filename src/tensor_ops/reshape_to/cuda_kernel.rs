use crate::{
    shapes::{HasSameNumelAs, Shape},
    tensor::cuda::{Cuda, CudaArray},
};
use cudarc::driver::{LaunchAsync, LaunchConfig};
use std::sync::Arc;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/reshape.ptx"));

macro_rules! impl_reshape {
    ($TypeName:ty, $Mod:tt, $Fwd:tt, $Bwd:tt) => {
        impl super::ReshapeKernel<$TypeName> for Cuda {
            fn forward<Src: Shape, Dst: Shape>(
                &self,
                dst: Dst,
                inp: &Self::Storage<Src, $TypeName>,
            ) -> Result<Self::Storage<Dst, $TypeName>, Self::Err>
            where
                Src: HasSameNumelAs<Dst>,
            {
                if !self.dev.has_func($Mod, $Fwd) {
                    self.dev.load_ptx(PTX_SRC.into(), $Mod, &[$Fwd, $Bwd])?;
                }

                let numel = inp.data.len();
                let mut storage = unsafe { self.dev.alloc_async::<$TypeName>(numel) }?;

                let inp_dims = self.dev.take_async(inp.shape.concrete().into())?;
                let dst_dims = self.dev.take_async(dst.concrete().into())?;
                let inp_strides = self.dev.take_async(inp.strides.into())?;
                let dst_strides = self.dev.take_async(dst.strides().into())?;

                let fwd_fn = self.dev.get_func($Mod, $Fwd).unwrap();
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
                grad_inp: &mut Self::Storage<Src, $TypeName>,
                grad_out: &Self::Storage<Dst, $TypeName>,
            ) -> Result<(), Self::Err>
            where
                Src: HasSameNumelAs<Dst>,
            {
                let bwd_fn = self.dev.get_func($Mod, $Bwd).unwrap();
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
    };
}

impl_reshape!(
    f32,
    "reshape_f32",
    "reshape_forward_f32",
    "reshape_backward_f32"
);
impl_reshape!(
    f64,
    "reshape_f64",
    "reshape_forward_f64",
    "reshape_backward_f64"
);
