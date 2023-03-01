use crate::{
    optim::optimizer::{Momentum, WeightDecay},
    shapes::Dtype,
    tensor::cpu::*,
};

use super::{SgdConfig, SgdKernel};

impl<E: Dtype> SgdKernel<E> for Cpu {
    fn update(
        &self,
        cfg: &SgdConfig<E>,
        param: &mut Self::Vec<E>,
        velocity: &mut Self::Vec<E>,
        grad: Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        for ((p, mut g), v) in param
            .iter_mut()
            .zip(grad.iter().cloned())
            .zip(velocity.iter_mut())
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

        Ok(())
    }
}
