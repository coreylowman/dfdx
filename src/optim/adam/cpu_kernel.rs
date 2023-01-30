use super::{AdamConfig, AdamKernel};
use crate::{
    optim::WeightDecay,
    shapes::{Dtype, Shape},
    tensor::Cpu,
};

impl<F: num_traits::Float + Dtype> AdamKernel<F> for Cpu {
    fn update<S: Shape>(
        &self,
        t: i32,
        cfg: &AdamConfig<F>,
        param: &mut Self::Storage<S, F>,
        moment1: &mut Self::Storage<S, F>,
        moment2: &mut Self::Storage<S, F>,
        grad: Self::Storage<S, F>,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(param.data.len(), grad.data.len());
        debug_assert_eq!(param.shape, grad.shape);
        debug_assert_eq!(param.strides, grad.strides);

        for ((p, mut g), (m, v)) in param
            .buf_iter_mut()
            .zip(grad.buf_iter().cloned())
            .zip(moment1.buf_iter_mut().zip(moment2.buf_iter_mut()))
        {
            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                g += wd * *p;
            }

            *m = *m * cfg.betas[0] + g * (F::one() - cfg.betas[0]);
            *v = *v * cfg.betas[1] + g.powi(2) * (F::one() - cfg.betas[1]);
            let m_hat = *m * (F::one() - cfg.betas[0].powi(t)).recip();
            let v_hat = *v * (F::one() - cfg.betas[1].powi(t)).recip();
            g = cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);

            if let Some(WeightDecay::Decoupled(wd)) = cfg.weight_decay {
                g += wd * cfg.lr * *p;
            }

            *p -= g;
        }
        Ok(())
    }
}
