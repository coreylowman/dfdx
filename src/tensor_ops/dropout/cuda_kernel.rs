use crate::{
    shapes::{Shape},
    tensor::cuda::{Cuda, CudaArray},
};

use std::{sync::Arc, vec::Vec};

use cudarc::driver::{LaunchAsync, LaunchConfig};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Standard;

const MODULE_NAME: &str = "dropout";
const FWD_FN_NAME: &str = "dropout_forward_f32";
const BWD_FN_NAME: &str = "dropout_backward_f32";
const ALL_FN_NAMES: [&str; 2] = [FWD_FN_NAME, BWD_FN_NAME];
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/dropout.ptx"));

impl super::DropoutKernel<f32> for Cuda {
    fn forward<S: Shape>(
        &self,
        op: super::DropoutKernelOp<f32>,
        inp: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, f32>, Self::Err> {
        let noise = {
            let mut rng = StdRng::seed_from_u64(op.seed);
            let mut noise: Vec<f32> = Vec::with_capacity(inp.data.len());
            noise.resize_with(inp.data.len(), || rng.sample(Standard));
            self.dev.take_async(noise)
        }?;

        if !self.dev.has_func(MODULE_NAME, FWD_FN_NAME) {
            self.dev
                .load_ptx(PTX_SRC.into(), MODULE_NAME, &ALL_FN_NAMES)?;
        }

        let numel = inp.data.len();
        let mut storage = self.dev.alloc_zeros_async::<f32>(numel)?;

        let fwd_fn = self.dev.get_func(MODULE_NAME, FWD_FN_NAME).unwrap();
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
        op: super::DropoutKernelOp<f32>,
        inp: &Self::Storage<S, f32>,
        grad_inp: &mut Self::Storage<S, f32>,
        grad_out: &Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        let noise = {
            let mut rng = StdRng::seed_from_u64(op.seed);
            let mut noise: Vec<f32> = Vec::with_capacity(inp.data.len());
            noise.resize_with(inp.data.len(), || rng.sample(Standard));
            self.dev.take_async(noise)
        }?;
        let bwd_fn = self.dev.get_func(MODULE_NAME, BWD_FN_NAME).unwrap();
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
