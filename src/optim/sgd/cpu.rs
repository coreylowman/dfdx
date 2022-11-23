use crate::{
    arrays::{Dtype, Shape},
    devices::cpu::*,
    optim::optimizer::{Momentum, WeightDecay},
};

use super::{SgdConfig, SgdUpdate};

impl<E: Dtype> SgdUpdate<Cpu, E> for SgdConfig<E> {
    fn update_param<S: Shape>(
        &self,
        param: &mut StridedArray<S, E>,
        velocity: &mut StridedArray<S, E>,
        grad: StridedArray<S, E>,
    ) {
        debug_assert_eq!(param.data.len(), grad.data.len());
        debug_assert_eq!(param.shape, grad.shape);
        debug_assert_eq!(param.strides, grad.strides);

        for ((p, mut g), v) in param
            .buf_iter_mut()
            .zip(grad.buf_iter().cloned())
            .zip(velocity.buf_iter_mut())
        {
            if let Some(WeightDecay::L2(wd)) = self.weight_decay {
                g += wd * *p;
            }

            match self.momentum {
                Some(Momentum::Classic(u)) => {
                    *v = g + u * *v;
                    g = *v * self.lr;
                }
                Some(Momentum::Nesterov(u)) => {
                    *v = g + u * *v;
                    g = (g + u * *v) * self.lr;
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
