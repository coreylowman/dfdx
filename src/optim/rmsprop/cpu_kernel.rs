use crate::{optim::WeightDecay, shapes::Dtype, tensor::cpu::Cpu};

use super::{RMSpropConfig, RMSpropKernel};

impl<E: num_traits::Float + Dtype> RMSpropKernel<E> for Cpu {
    fn update(
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
