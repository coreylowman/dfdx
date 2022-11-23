use crate::{
    devices::cpu::{Cpu, StridedArray},
    optim::WeightDecay,
};

use super::{RMSpropConfig, RMSpropUpdate};

impl RMSpropUpdate<Cpu, f32> for RMSpropConfig<f32> {
    fn update_param<S: crate::arrays::Shape>(
        &self,
        param: &mut StridedArray<S, f32>,
        momentum: &mut StridedArray<S, f32>,
        square_avg: &mut StridedArray<S, f32>,
        grad_avg: &mut StridedArray<S, f32>,
        grad: StridedArray<S, f32>,
    ) {
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
            if let Some(WeightDecay::L2(wd)) = self.weight_decay {
                g += wd * *p;
            }

            // sa = a * sa + (1 - a) * g^2
            *s_avg += (1.0 - self.alpha) * (g * g - *s_avg);

            let avg = if self.centered {
                // ga = a * ga + (1 - a) * g
                *g_avg += (1.0 - self.alpha) * (g - *g_avg);
                // NOTE: self.eps in sqrt
                (*s_avg - g_avg.powi(2) + self.eps).sqrt()
            } else {
                // NOTE: self.eps in sqrt
                (*s_avg + self.eps).sqrt()
            };

            g /= avg;

            match self.momentum {
                Some(u) => {
                    *m = *m * u + g;
                    g = *m * self.lr;
                }
                None => g *= self.lr,
            }

            if let Some(WeightDecay::Decoupled(wd)) = self.weight_decay {
                g += wd * self.lr * *p;
            }

            *p -= g;
        }
    }
}
