use crate::{
    dtypes::{Dtype, NotMixedPrecision},
    tensor::cpu::Cpu,
};

use super::{RMSpropConfig, RMSpropKernel, WeightDecay};

#[cfg(feature = "f16")]
impl RMSpropKernel<crate::dtypes::AMP<crate::dtypes::f16>> for Cpu {
    fn rmsprop_kernel(
        &self,
        cfg: &RMSpropConfig,
        param: &mut Self::Vec,
        momentum: &mut Self::Vec,
        square_avg: &mut Self::Vec,
        grad_avg: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let alpha = cfg.alpha as f32;
        let eps = cfg.eps as f32;
        let lr = cfg.lr as f32;

        for ((p, g), (s_avg, (g_avg, m))) in param.iter_mut().zip(grad.iter().cloned()).zip(
            square_avg
                .iter_mut()
                .zip(grad_avg.iter_mut().zip(momentum.iter_mut())),
        ) {
            let p_f32 = p.0.to_f32();
            let mut g_f32 = g.0.to_f32();
            let mut s_avg_f32 = s_avg.0.to_f32();
            let mut g_avg_f32 = g_avg.0.to_f32();
            let mut m_f32 = m.0.to_f32();

            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                g_f32 += wd as f32 * p_f32;
            }

            // sa = a * sa + (1 - a) * g^2
            s_avg_f32 += (1.0 - alpha) * (g_f32 * g_f32 - s_avg_f32);

            let avg = if cfg.centered {
                // ga = a * ga + (1 - a) * g
                g_avg_f32 += (1.0 - alpha) * (g_f32 - g_avg_f32);
                // NOTE: eps in sqrt
                (s_avg_f32 - g_avg_f32.powi(2) + eps).sqrt()
            } else {
                // NOTE: eps in sqrt
                (s_avg_f32 + eps).sqrt()
            };

            g_f32 /= avg;

            match cfg.momentum {
                Some(u) => {
                    m_f32 = m_f32 * (u as f32) + g_f32;
                    g_f32 = m_f32 * lr;
                }
                None => g_f32 *= lr,
            }

            if let Some(WeightDecay::Decoupled(wd)) = cfg.weight_decay {
                g_f32 += (wd * cfg.lr) as f32 * p_f32;
            }

            p.0 = crate::dtypes::f16::from_f32(p_f32 - g_f32);
            s_avg.0 = crate::dtypes::f16::from_f32(s_avg_f32);
            g_avg.0 = crate::dtypes::f16::from_f32(g_avg_f32);
            m.0 = crate::dtypes::f16::from_f32(m_f32);
        }
        Ok(())
    }
}

impl<E: num_traits::Float + Dtype + NotMixedPrecision> RMSpropKernel<E> for Cpu {
    fn rmsprop_kernel(
        &self,
        cfg: &RMSpropConfig,
        param: &mut Self::Vec,
        momentum: &mut Self::Vec,
        square_avg: &mut Self::Vec,
        grad_avg: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let alpha = E::from_f64(cfg.alpha).unwrap();
        let eps = E::from_f64(cfg.eps).unwrap();
        let lr = E::from_f64(cfg.lr).unwrap();
        for ((p, mut g), (s_avg, (g_avg, m))) in param.iter_mut().zip(grad.iter().cloned()).zip(
            square_avg
                .iter_mut()
                .zip(grad_avg.iter_mut().zip(momentum.iter_mut())),
        ) {
            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                g += E::from_f64(wd).unwrap() * *p;
            }

            // sa = a * sa + (1 - a) * g^2
            *s_avg += (E::one() - alpha) * (g * g - *s_avg);

            let avg = if cfg.centered {
                // ga = a * ga + (1 - a) * g
                *g_avg += (E::one() - alpha) * (g - *g_avg);
                // NOTE: eps in sqrt
                (*s_avg - g_avg.powi(2) + eps).sqrt()
            } else {
                // NOTE: eps in sqrt
                (*s_avg + eps).sqrt()
            };

            g /= avg;

            match cfg.momentum {
                Some(u) => {
                    *m = *m * E::from_f64(u).unwrap() + g;
                    g = *m * lr;
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
