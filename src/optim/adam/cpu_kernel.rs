use super::{AdamConfig, AdamKernel};
use crate::{optim::WeightDecay, shapes::Dtype, tensor::Cpu};

impl<E: num_traits::Float + Dtype> AdamKernel<E> for Cpu {
    fn update(
        &self,
        t: i32,
        cfg: &AdamConfig,
        param: &mut Self::Vec<E>,
        moment1: &mut Self::Vec<E>,
        moment2: &mut Self::Vec<E>,
        grad: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let betas = cfg.betas.map(E::from_f64).map(Option::unwrap);
        let eps = E::from_f64(cfg.eps).unwrap();
        let lr = E::from_f64(cfg.lr).unwrap();

        for ((p, mut g), (m, v)) in param
            .iter_mut()
            .zip(grad.iter().cloned())
            .zip(moment1.iter_mut().zip(moment2.iter_mut()))
        {
            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                g += E::from_f64(wd).unwrap() * *p;
            }

            *m = *m * betas[0] + g * (E::one() - betas[0]);
            *v = *v * betas[1] + g.powi(2) * (E::one() - betas[1]);
            let m_hat = *m * (E::one() - betas[0].powi(t)).recip();
            let v_hat = *v * (E::one() - betas[1].powi(t)).recip();
            g = lr * m_hat / (v_hat.sqrt() + eps);

            if let Some(WeightDecay::Decoupled(wd)) = cfg.weight_decay {
                g += E::from_f64(wd * cfg.lr).unwrap() * *p;
            }

            *p -= g;
        }
        Ok(())
    }
}
