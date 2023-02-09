use crate::{
    shapes::Shape,
    tensor::cuda::{Cuda, CudaArray},
};

use std::{sync::Arc, vec::Vec};

use cudarc::driver::{LaunchAsync, LaunchConfig};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Standard;

const MODULE_NAME: &str = "dropout";
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/dropout.ptx"));

macro_rules! impl_dropout {
    ($TypeName:ty, $ForwardFn:tt, $BackwardFn:tt) => {
        impl super::DropoutKernel<$TypeName> for Cuda {
            fn forward<S: Shape>(
                &self,
                op: super::DropoutKernelOp<$TypeName>,
                inp: &Self::Storage<S, $TypeName>,
            ) -> Result<Self::Storage<S, $TypeName>, Self::Err> {
                let noise = {
                    let mut rng = StdRng::seed_from_u64(op.seed);
                    let mut noise: Vec<$TypeName> = Vec::with_capacity(inp.data.len());
                    noise.resize_with(inp.data.len(), || rng.sample(Standard));
                    self.dev.take_async(noise)
                }?;

                if !self.dev.has_func(MODULE_NAME, $ForwardFn) {
                    self.dev
                        .load_ptx(PTX_SRC.into(), MODULE_NAME, &[$ForwardFn, $BackwardFn])?;
                }

                let numel = inp.data.len();
                let mut storage = unsafe { self.dev.alloc_async::<$TypeName>(numel) }?;

                let fwd_fn = self.dev.get_func(MODULE_NAME, $ForwardFn).unwrap();
                let cfg = LaunchConfig::for_num_elems(numel as u32);
                let params = (
                    op.prob,           // const float prob,
                    numel,             // const size_t numel,
                    inp.data.as_ref(), // const float *inp,
                    &noise,            // const float *noise,
                    &mut storage,      // float *out
                );
                unsafe { fwd_fn.launch_async(cfg, params) }?;
                Ok(CudaArray {
                    data: Arc::new(storage),
                    shape: inp.shape,
                    strides: inp.strides,
                })
            }
            fn backward<S: Shape>(
                &self,
                op: super::DropoutKernelOp<$TypeName>,
                inp: &Self::Storage<S, $TypeName>,
                grad_inp: &mut Self::Storage<S, $TypeName>,
                grad_out: &Self::Storage<S, $TypeName>,
            ) -> Result<(), Self::Err> {
                let noise = {
                    let mut rng = StdRng::seed_from_u64(op.seed);
                    let mut noise: Vec<$TypeName> = Vec::with_capacity(inp.data.len());
                    noise.resize_with(inp.data.len(), || rng.sample(Standard));
                    self.dev.take_async(noise)
                }?;
                let bwd_fn = self.dev.get_func(MODULE_NAME, $BackwardFn).unwrap();
                let numel = inp.data.len();
                let cfg = LaunchConfig::for_num_elems(numel as u32);
                let params = (
                    op.prob,                           // const float prob,
                    numel,                             // const size_t numel,
                    &noise,                            // const float *noise,
                    Arc::make_mut(&mut grad_inp.data), // float *grad_inp,
                    grad_out.data.as_ref(),            // const float *grad_out
                );
                unsafe { bwd_fn.launch_async(cfg, params) }?;
                Ok(())
            }
        }
    };
}

impl_dropout!(f32, "dropout_forward_f32", "dropout_backward_f32");
impl_dropout!(f64, "dropout_forward_f64", "dropout_backward_f64");
