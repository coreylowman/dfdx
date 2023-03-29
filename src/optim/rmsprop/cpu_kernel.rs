use crate::{optim::WeightDecay, shapes::Dtype, tensor::cpu::Cpu};

use super::{RMSpropConfig, RMSpropKernel};

impl<E: num_traits::Float + Dtype> RMSpropKernel<E> for Cpu {
    fn update(
        &self,
        cfg: &RMSpropConfig<E>,
        param: &mut Self::Storage,
        momentum: &mut Self::Storage,
        square_avg: &mut Self::Storage,
        grad_avg: &mut Self::Storage,
        grad: &Self::Storage,
    ) -> Result<(), Self::Err> {
        for ((p, mut g), (s_avg, (g_avg, m))) in param.iter_mut().zip(grad.iter().cloned()).zip(
            square_avg
                .iter_mut()
                .zip(grad_avg.iter_mut().zip(momentum.iter_mut())),
        ) {
            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                g += wd * *p;
            }

            // sa = a * sa + (1 - a) * g^2
            *s_avg += (E::one() - cfg.alpha) * (g * g - *s_avg);

            let avg = if cfg.centered {
                // ga = a * ga + (1 - a) * g
                *g_avg += (E::one() - cfg.alpha) * (g - *g_avg);
                // NOTE: cfg.eps in sqrt
                (*s_avg - g_avg.powi(2) + cfg.eps).sqrt()
            } else {
                // NOTE: cfg.eps in sqrt
                (*s_avg + cfg.eps).sqrt()
            };

            g /= avg;

            match cfg.momentum {
                Some(u) => {
                    *m = *m * u + g;
                    g = *m * cfg.lr;
                }
                None => g *= cfg.lr,
            }

            if let Some(WeightDecay::Decoupled(wd)) = cfg.weight_decay {
                g += wd * cfg.lr * *p;
            }

            *p -= g;
        }
        Ok(())
    }
}
