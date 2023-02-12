use crate::{
    shapes::*,
    tensor::cuda::{Cuda, CudaArray},
};

use std::{sync::Arc, vec::Vec};

use cudarc::driver::{AsKernelParam, LaunchAsync, LaunchConfig};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Standard};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/dropout.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "dropout_f32";
    const FNS: &'static [&'static str] = &["dropout_forward_f32", "dropout_backward_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "dropout_f64";
    const FNS: &'static [&'static str] = &["dropout_forward_f64", "dropout_backward_f64"];
}

impl<E: Dtype + AsKernelParam> super::DropoutKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
    Standard: Distribution<E>,
{
    fn forward<S: Shape>(
        &self,
        op: super::DropoutKernelOp<E>,
        inp: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        let noise = {
            let mut rng = StdRng::seed_from_u64(op.seed);
            let mut noise: Vec<E> = Vec::with_capacity(inp.data.len());
            noise.resize_with(inp.data.len(), || rng.sample(Standard));
            self.dev.take_async(noise)
        }?;

        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let numel = inp.data.len();
        let mut storage = unsafe { self.dev.alloc_async::<E>(numel) }?;

        let fwd_fn = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();
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
        op: super::DropoutKernelOp<E>,
        inp: &Self::Storage<S, E>,
        grad_inp: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        let noise = {
            let mut rng = StdRng::seed_from_u64(op.seed);
            let mut noise: Vec<E> = Vec::with_capacity(inp.data.len());
            noise.resize_with(inp.data.len(), || rng.sample(Standard));
            self.dev.take_async(noise)
        }?;
        let bwd_fn = self.dev.get_func(Self::MOD, Self::FNS[1]).unwrap();
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
