use super::{AdamConfig, AdamKernel, WeightDecay};
use crate::{
    dtypes::{Dtype, NotMixedPrecision},
    tensor::Cpu,
};

#[cfg(feature = "f16")]
impl AdamKernel<crate::dtypes::AMP<crate::dtypes::f16>> for Cpu {
    fn adam_kernel(
        &self,
        t: i32,
        cfg: &AdamConfig,
        param: &mut Self::Vec,
        moment1: &mut Self::Vec,
        moment2: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let betas = cfg.betas.map(|x| x as f32);
        let eps = cfg.eps as f32;
        let lr = cfg.lr as f32;

        for ((p, g), (m, v)) in param
            .iter_mut()
            .zip(grad.iter().cloned())
            .zip(moment1.iter_mut().zip(moment2.iter_mut()))
        {
            let p_f32 = p.0.to_f32();
            let mut g_f32 = g.0.to_f32();
            let mut m_f32 = m.0.to_f32();
            let mut v_f32 = v.0.to_f32();

            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                g_f32 += (wd as f32) * p_f32;
            }

            m_f32 = m_f32 * betas[0] + g_f32 * (1.0 - betas[0]);
            v_f32 = v_f32 * betas[1] + g_f32.powi(2) * (1.0 - betas[1]);
            let m_hat = m_f32 * (1.0 - betas[0].powi(t)).recip();
            let v_hat = v_f32 * (1.0 - betas[1].powi(t)).recip();
            g_f32 = lr * m_hat / (v_hat.sqrt() + eps);

            if let Some(WeightDecay::Decoupled(wd)) = cfg.weight_decay {
                g_f32 += (wd * cfg.lr) as f32 * p_f32;
            }

            p.0 = crate::dtypes::f16::from_f32(p_f32 - g_f32);
            m.0 = crate::dtypes::f16::from_f32(m_f32);
            v.0 = crate::dtypes::f16::from_f32(v_f32);
        }
        Ok(())
    }
}

impl<E: num_traits::Float + Dtype + NotMixedPrecision> AdamKernel<E> for Cpu {
    fn adam_kernel(
        &self,
        t: i32,
        cfg: &AdamConfig,
        param: &mut Self::Vec,
        moment1: &mut Self::Vec,
        moment2: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let betas = cfg.betas.map(E::from_f64).map(Option::unwrap);
        let eps = E::from_f64(cfg.eps).unwrap();
        let lr = E::from_f64(cfg.lr).unwrap();

        for ((p, mut g), (m, v)) in param
            .iter_mut()
            .zip(grad.iter().cloned())
            .zip(moment1.iter_mut().zip(moment2.iter_mut()))
        {
            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                g += E::from_f64(wd).unwrap() * *p;
            }

            *m = *m * betas[0] + g * (E::one() - betas[0]);
            *v = *v * betas[1] + g.powi(2) * (E::one() - betas[1]);
            let m_hat = *m * (E::one() - betas[0].powi(t)).recip();
            let v_hat = *v * (E::one() - betas[1].powi(t)).recip();
            g = lr * m_hat / (v_hat.sqrt() + eps);

            if let Some(WeightDecay::Decoupled(wd)) = cfg.weight_decay {
                g += E::from_f64(wd * cfg.lr).unwrap() * *p;
            }

            *p -= g;
        }
        Ok(())
    }
}
