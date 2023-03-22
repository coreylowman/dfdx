use crate::{
    shapes::*,
    tensor::{launch_cfg, Cuda, Tensor},
};

use std::vec::Vec;

use cudarc::driver::{DeviceSlice, LaunchAsync};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Standard};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/dropout.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "dropout_f32";
    const FNS: &'static [&'static str] = &["dropout_fwd_f32", "dropout_bwd_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "dropout_f64";
    const FNS: &'static [&'static str] = &["dropout_fwd_f64", "dropout_bwd_f64"];
}

impl<E: Dtype> super::DropoutKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
    Standard: Distribution<E>,
{
    fn forward<S: Shape>(
        &self,
        op: super::DropoutKernelOp<E>,
        inp: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let noise = {
            let mut rng = StdRng::seed_from_u64(op.seed);
            let mut noise: Vec<E> = Vec::with_capacity(inp.data.len());
            noise.resize_with(inp.data.len(), || rng.sample(Standard));
            self.dev.htod_copy(noise)
        }?;

        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let numel = inp.data.len();
        let mut storage = unsafe { self.dev.alloc::<E>(numel) }?;

        let fwd_fn = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();
        let cfg = launch_cfg(numel as u32);
        let params = (op.prob, numel, inp.data.as_ref(), &noise, &mut storage);
        unsafe { fwd_fn.launch(cfg, params) }?;
        Ok(self.build_tensor(inp.shape, inp.strides, storage))
    }
    fn backward<S: Shape>(
        &self,
        op: super::DropoutKernelOp<E>,
        inp: &Tensor<S, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let noise = {
            let mut rng = StdRng::seed_from_u64(op.seed);
            let mut noise: Vec<E> = Vec::with_capacity(inp.data.len());
            noise.resize_with(inp.data.len(), || rng.sample(Standard));
            self.dev.htod_copy(noise)
        }?;
        let bwd_fn = self.dev.get_func(Self::MOD, Self::FNS[1]).unwrap();
        let numel = inp.data.len();
        let cfg = launch_cfg(numel as u32);
        let params = (op.prob, numel, &noise, grad_inp, grad_out);
        unsafe { bwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}
