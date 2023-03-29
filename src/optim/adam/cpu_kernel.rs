use super::{AdamConfig, AdamKernel};
use crate::{optim::WeightDecay, shapes::Dtype, tensor::Cpu};

impl<E: num_traits::Float + Dtype> AdamKernel<E> for Cpu {
    fn update(
        &self,
        t: i32,
        cfg: &AdamConfig<E>,
        param: &mut Self::Storage,
        moment1: &mut Self::Storage,
        moment2: &mut Self::Storage,
        grad: &Self::Storage,
    ) -> Result<(), Self::Err> {
        for ((p, mut g), (m, v)) in param
            .iter_mut()
            .zip(grad.iter().cloned())
            .zip(moment1.iter_mut().zip(moment2.iter_mut()))
        {
            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                g += wd * *p;
            }

            *m = *m * cfg.betas[0] + g * (E::one() - cfg.betas[0]);
            *v = *v * cfg.betas[1] + g.powi(2) * (E::one() - cfg.betas[1]);
            let m_hat = *m * (E::one() - cfg.betas[0].powi(t)).recip();
            let v_hat = *v * (E::one() - cfg.betas[1].powi(t)).recip();
            g = cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);

            if let Some(WeightDecay::Decoupled(wd)) = cfg.weight_decay {
                g += wd * cfg.lr * *p;
            }

            *p -= g;
        }
        Ok(())
    }
}
