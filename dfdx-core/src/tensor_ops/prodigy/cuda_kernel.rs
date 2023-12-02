use crate::{
    dtypes::*,
    prelude::Storage,
    tensor::{launch_cfg, Cuda, Error},
    tensor_ops::optim::*,
};

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync};

#[repr(C)]
#[derive(Clone)]
struct CudaProdigyConfig1 {
    numel: usize,
    k: i32,
    lr: f64,
    beta1: f64,
    beta2: f64,
    beta3: f64,
    weight_decay_type: WeightDecayType,
    weight_decay: f64,
    bias_correction: f64,
    safeguard_warmup: bool,
    d0: f64,
}

#[repr(C)]
#[derive(Clone)]
struct CudaProdigyConfig2 {
    numel: usize,
    lr: f64,
    eps: f64,
    weight_decay_type: WeightDecayType,
    weight_decay: f64,
    bias_correction: f64,
}

unsafe impl DeviceRepr for CudaProdigyConfig1 {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut _
    }
}

unsafe impl DeviceRepr for CudaProdigyConfig2 {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut _
    }
}

impl CudaProdigyConfig1 {
    fn new(cfg: &super::ProdigyConfig, numel: usize, k: i32) -> Self {
        let [beta1, beta2] = cfg.betas;
        let beta3 = if let Some(beta3) = cfg.beta3 {
            beta3
        } else {
            beta2.sqrt()
        };
        let (weight_decay_type, weight_decay) = weight_decay_to_cuda(cfg.weight_decay);

        let bias_correction = if cfg.use_bias_correction {
            (1.0 - beta2.powi(k)).sqrt() / (1.0 - beta1.powi(k))
        } else {
            1.
        };

        CudaProdigyConfig1 {
            numel,
            k,
            lr: cfg.lr,
            beta1,
            beta2,
            beta3,
            weight_decay_type,
            weight_decay,
            bias_correction,
            safeguard_warmup: cfg.safeguard_warmup,
            d0: cfg.d0,
        }
    }
}

impl CudaProdigyConfig2 {
    fn new(cfg: &super::ProdigyConfig, cfg1: &CudaProdigyConfig1) -> Self {
        CudaProdigyConfig2 {
            numel: cfg1.numel,
            lr: cfg1.lr,
            eps: cfg.eps,
            weight_decay_type: cfg1.weight_decay_type,
            weight_decay: cfg1.weight_decay,
            bias_correction: cfg1.bias_correction,
        }
    }
}

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/prodigy.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FWD1: &'static str;
    const FWD2: &'static str;
}

#[cfg(feature = "f16")]
impl HasCudaKernel<AMP<f16>> for Cuda {
    const MOD: &'static str = "prodigy_amp_f16";
    const FWD1: &'static str = "prodigy_update1_amp_f16";
    const FWD2: &'static str = "prodigy_update2_amp_f16";
}

#[cfg(feature = "f16")]
impl HasCudaKernel<f16> for Cuda {
    const MOD: &'static str = "prodigy_f16";
    const FWD1: &'static str = "prodigy_update1_f16";
    const FWD2: &'static str = "prodigy_update2_f16";
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "prodigy_f32";
    const FWD1: &'static str = "prodigy_update1_f32";
    const FWD2: &'static str = "prodigy_update2_f32";
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "prodigy_f64";
    const FWD1: &'static str = "prodigy_update1_f64";
    const FWD2: &'static str = "prodigy_update2_f64";
}

impl<E: Dtype + num_traits::Zero> super::ProdigyKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn prodigy_kernel(
        &self,
        k: i32,
        d: &mut f64,
        d_max: &mut f64,
        d_numerator: &mut f64,
        cfg: &super::ProdigyConfig,
        param: &mut Self::Vec,
        s: &mut Self::Vec,
        p0: &mut Self::Vec,
        p0b: &mut Self::Vec,
        moment1: &mut Self::Vec,
        moment2: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Error> {
        if !self.dev.has_func(Self::MOD, Self::FWD1) {
            self.dev
                .load_ptx(PTX_SRC.into(), Self::MOD, &[Self::FWD1, Self::FWD2])?;
        }

        let numel = param.len();
        let opt_cfg1 = CudaProdigyConfig1::new(cfg, numel, k);
        let opt_cfg2 = CudaProdigyConfig2::new(cfg, &opt_cfg1);
        let func1 = self.dev.get_func(Self::MOD, Self::FWD1).unwrap();
        let cu_cfg = launch_cfg::<128>(numel as u32);

        // d_numerators for thread-block sum-reduction
        let mut d_numerators: Self::Vec =
            self.try_alloc_len(cu_cfg.grid_dim.0 as usize * cu_cfg.block_dim.0 as usize)?;
        let mut d_numerators_vec =
            vec![E::zero(); cu_cfg.grid_dim.0 as usize * cu_cfg.block_dim.0 as usize];
        self.dev
            .htod_sync_copy_into(d_numerators_vec.as_slice(), &mut d_numerators)?;
        // d_denom for thread-block sum-reduction
        let mut d_denoms: Self::Vec =
            self.try_alloc_len(cu_cfg.grid_dim.0 as usize * cu_cfg.block_dim.0 as usize)?;
        let mut d_denoms_vec =
            vec![E::zero(); cu_cfg.grid_dim.0 as usize * cu_cfg.block_dim.0 as usize];
        self.dev
            .htod_sync_copy_into(d_denoms_vec.as_slice(), &mut d_denoms)?;

        // local cache
        let d_old = *d;
        let beta3 = opt_cfg1.beta3;

        let params1 = (
            opt_cfg1,
            *d,
            //
            &mut d_numerators,
            &mut d_denoms,
            //
            &*param,
            s,
            p0,
            p0b,
            &mut *moment1,
            &mut *moment2,
            grad,
        );
        unsafe { func1.launch(cu_cfg.clone(), params1) }?;

        // get the thread-block d_numerators and d_denoms
        self.dev
            .dtoh_sync_copy_into(&d_numerators.data, d_numerators_vec.as_mut_slice())?;
        self.dev
            .dtoh_sync_copy_into(&d_denoms.data, d_denoms_vec.as_mut_slice())?;
        // sum and update d_numerators and d_denoms
        let d_numerator_: E = d_numerators_vec
            .into_iter()
            .reduce(|acc, e| acc + e)
            .unwrap()
            + E::from_f64(*d_numerator).unwrap() * E::from_f64(beta3).unwrap();
        let d_denom_: E = d_denoms_vec.into_iter().reduce(|acc, e| acc + e).unwrap();

        if d_denom_ == E::zero() {
            return Ok(());
        }

        let func2 = self.dev.get_func(Self::MOD, Self::FWD2).unwrap();

        *d_numerator = d_numerator_.to_f64().unwrap();
        let global_d_denom = d_denom_.to_f64().unwrap();
        if cfg.lr > 0. {
            let d_hat = cfg.d_coef * *d_numerator / global_d_denom;
            if *d == cfg.d0 {
                *d = d.max(d_hat);
            }
            *d_max = d_max.max(d_hat);
            *d = d_max.min(*d * cfg.growth_rate);
        }

        let params2 = (opt_cfg2, d_old, param, &*moment1, &*moment2);
        unsafe { func2.launch(cu_cfg, params2) }?;

        Ok(())
    }
}
