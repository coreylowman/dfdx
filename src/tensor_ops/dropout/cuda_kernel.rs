use crate::{
    dtypes::*,
    shapes::*,
    tensor::{launch_cfg, Cuda, Tensor},
};

use std::vec::Vec;

use cudarc::driver::{DeviceSlice, LaunchAsync};

use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Bernoulli, Distribution};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/dropout.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

#[cfg(feature = "f16")]
impl HasCudaKernel<f16> for Cuda {
    const MOD: &'static str = "dropout_f16";
    const FNS: &'static [&'static str] = &["dropout_fwd_f16", "dropout_bwd_f16"];
}

#[cfg(feature = "f16")]
impl HasCudaKernel<AMP<f16>> for Cuda {
    const MOD: &'static str = "dropout_f16";
    const FNS: &'static [&'static str] = &["dropout_fwd_f16", "dropout_bwd_f16"];
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
{
    fn forward<S: Shape>(
        &self,
        op: super::DropoutKernelOp,
        inp: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let mask = {
            let mut rng = StdRng::seed_from_u64(op.seed);
            let dist = Bernoulli::new(op.prob).unwrap();
            let mut mask: Vec<bool> = Vec::with_capacity(inp.data.len());
            mask.resize_with(inp.data.len(), || dist.sample(&mut rng));
            self.dev.htod_copy(mask)
        }?;

        let prob = E::from_f64(op.prob).unwrap();

        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let numel = inp.data.len();
        let mut storage = unsafe { self.alloc_empty::<E>(numel) }?;

        let fwd_fn = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();
        let cfg = launch_cfg::<128>(numel as u32);
        let params = (prob, numel, inp.data.as_ref(), &mask, &mut storage);
        unsafe { fwd_fn.launch(cfg, params) }?;
        Ok(self.build_tensor(inp.shape, inp.strides, storage))
    }
    fn backward<S: Shape>(
        &self,
        op: super::DropoutKernelOp,
        inp: &Tensor<S, E, Self>,
        grad_inp: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let mask = {
            let mut rng = StdRng::seed_from_u64(op.seed);
            let dist = Bernoulli::new(op.prob).unwrap();
            let mut mask: Vec<bool> = Vec::with_capacity(inp.data.len());
            mask.resize_with(inp.data.len(), || dist.sample(&mut rng));
            self.dev.htod_copy(mask)
        }?;
        let prob = E::from_f64(op.prob).unwrap();
        let bwd_fn = self.dev.get_func(Self::MOD, Self::FNS[1]).unwrap();
        let numel = inp.data.len();
        let cfg = launch_cfg::<128>(numel as u32);
        let params = (prob, numel, &mask, grad_inp, grad_out);
        unsafe { bwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}
