use crate::{
    optim::optimizer::{Momentum, WeightDecay},
    shapes::{Dtype, Shape},
    tensor::cpu::*,
};

use super::{SgdConfig, SgdKernel};

impl<E: Dtype> SgdKernel<E> for Cpu {
    fn update<S: Shape>(
        cfg: &SgdConfig<E>,
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
            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                g += wd * *p;
            }

            match cfg.momentum {
                Some(Momentum::Classic(u)) => {
                    *v = g + u * *v;
                    g = *v * cfg.lr;
                }
                Some(Momentum::Nesterov(u)) => {
                    *v = g + u * *v;
                    g = (g + u * *v) * cfg.lr;
                }
                None => g *= cfg.lr,
            }

            if let Some(WeightDecay::Decoupled(wd)) = cfg.weight_decay {
                g += wd * cfg.lr * *p;
            }

            *p -= g;
        }
    }
}
