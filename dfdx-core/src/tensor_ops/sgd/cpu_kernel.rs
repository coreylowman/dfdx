use crate::{
    dtypes::{Dtype, NotMixedPrecision},
    tensor::cpu::*,
};

use super::{Momentum, SgdConfig, SgdKernel, WeightDecay};

#[cfg(feature = "f16")]
impl SgdKernel<crate::dtypes::AMP<crate::dtypes::f16>> for Cpu {
    fn sgd_kernel(
        &self,
        cfg: &SgdConfig,
        param: &mut Self::Vec,
        velocity: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let lr = cfg.lr as f32;

        for ((p, g), v) in param
            .iter_mut()
            .zip(grad.iter().cloned())
            .zip(velocity.iter_mut())
        {
            let p_f32 = p.0.to_f32();
            let mut g_f32 = g.0.to_f32();
            let mut v_f32 = v.0.to_f32();

            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                g_f32 += (wd as f32) * p_f32;
            }

            match cfg.momentum {
                Some(Momentum::Classic(u)) => {
                    let u = u as f32;
                    v_f32 = g_f32 + u * v_f32;
                    g_f32 = v_f32 * lr;
                }
                Some(Momentum::Nesterov(u)) => {
                    let u = u as f32;
                    v_f32 = g_f32 + u * v_f32;
                    g_f32 = (g_f32 + u * v_f32) * lr;
                }
                None => g_f32 *= lr,
            }

            if let Some(WeightDecay::Decoupled(wd)) = cfg.weight_decay {
                g_f32 += (wd * cfg.lr) as f32 * p_f32;
            }

            p.0 = crate::dtypes::f16::from_f32(p_f32 - g_f32);
            v.0 = crate::dtypes::f16::from_f32(v_f32);
        }

        Ok(())
    }
}

impl<E: Dtype + NotMixedPrecision> SgdKernel<E> for Cpu {
    fn sgd_kernel(
        &self,
        cfg: &SgdConfig,
        param: &mut Self::Vec,
        velocity: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let lr = E::from_f64(cfg.lr).unwrap();

        for ((p, mut g), v) in param
            .iter_mut()
            .zip(grad.iter().cloned())
            .zip(velocity.iter_mut())
        {
            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                let wd = E::from_f64(wd).unwrap();
                g += wd * *p;
            }

            match cfg.momentum {
                Some(Momentum::Classic(u)) => {
                    let u = E::from_f64(u).unwrap();
                    *v = g + u * *v;
                    g = *v * lr;
                }
                Some(Momentum::Nesterov(u)) => {
                    let u = E::from_f64(u).unwrap();
                    *v = g + u * *v;
                    g = (g + u * *v) * lr;
                }
                None => g *= lr,
            }

            if let Some(WeightDecay::Decoupled(wd)) = cfg.weight_decay {
                g += E::from_f64(wd * cfg.lr).unwrap() * *p;
            }

            *p -= g;
        }

        Ok(())
    }
}
