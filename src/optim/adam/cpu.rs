use crate::{
    arrays::Shape,
    devices::{cpu::StridedArray, Cpu},
    optim::WeightDecay,
};

use super::{AdamConfig, AdamUpdate};

impl AdamUpdate<Cpu, f32> for AdamConfig<f32> {
    fn update_param<S: Shape>(
        &self,
        t: i32,
        param: &mut StridedArray<S, f32>,
        moment1: &mut StridedArray<S, f32>,
        moment2: &mut StridedArray<S, f32>,
        grad: StridedArray<S, f32>,
    ) {
        debug_assert_eq!(param.data.len(), grad.data.len());
        debug_assert_eq!(param.shape, grad.shape);
        debug_assert_eq!(param.strides, grad.strides);

        for ((p, mut g), (m, v)) in param
            .buf_iter_mut()
            .zip(grad.buf_iter().cloned())
            .zip(moment1.buf_iter_mut().zip(moment2.buf_iter_mut()))
        {
            if let Some(WeightDecay::L2(wd)) = self.weight_decay {
                g += wd * *p;
            }

            *m = *m * self.betas[0] + g * (1.0 - self.betas[0]);
            *v = *v * self.betas[1] + g.powi(2) * (1.0 - self.betas[1]);
            let m_hat = *m * (1.0 - self.betas[0].powi(t)).recip();
            let v_hat = *v * (1.0 - self.betas[1].powi(t)).recip();
            g = self.lr * m_hat / (v_hat.sqrt() + self.eps);

            if let Some(WeightDecay::Decoupled(wd)) = self.weight_decay {
                g += wd * self.lr * *p;
            }

            *p -= g;
        }
    }
}
