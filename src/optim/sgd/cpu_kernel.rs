use crate::{
    optim::optimizer::{Momentum, WeightDecay},
    shapes::Dtype,
    tensor::cpu::*,
};

use super::{SgdConfig, SgdKernel};

impl<E: Dtype> SgdKernel<E> for Cpu {
    fn update(
        &self,
        cfg: &SgdConfig,
        param: &mut Self::Vec<E>,
        velocity: &mut Self::Vec<E>,
        grad: &Self::Vec<E>,
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
