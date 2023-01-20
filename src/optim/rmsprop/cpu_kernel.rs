use crate::{
    optim::WeightDecay,
    tensor::cpu::{Cpu, StridedArray},
};

use super::{RMSpropConfig, RMSpropKernel};

impl RMSpropKernel<f32> for Cpu {
    fn update<S: crate::shapes::Shape>(
        &self,
        cfg: &RMSpropConfig<f32>,
        param: &mut StridedArray<S, f32>,
        momentum: &mut StridedArray<S, f32>,
        square_avg: &mut StridedArray<S, f32>,
        grad_avg: &mut StridedArray<S, f32>,
        grad: StridedArray<S, f32>,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(param.data.len(), grad.data.len());
        debug_assert_eq!(param.shape, grad.shape);
        debug_assert_eq!(param.strides, grad.strides);

        for ((p, mut g), (s_avg, (g_avg, m))) in
            param.buf_iter_mut().zip(grad.buf_iter().cloned()).zip(
                square_avg
                    .buf_iter_mut()
                    .zip(grad_avg.buf_iter_mut().zip(momentum.buf_iter_mut())),
            )
        {
            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                g += wd * *p;
            }

            // sa = a * sa + (1 - a) * g^2
            *s_avg += (1.0 - cfg.alpha) * (g * g - *s_avg);

            let avg = if cfg.centered {
                // ga = a * ga + (1 - a) * g
                *g_avg += (1.0 - cfg.alpha) * (g - *g_avg);
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
