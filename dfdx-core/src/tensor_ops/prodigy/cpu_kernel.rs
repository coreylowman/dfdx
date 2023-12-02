use super::super::WeightDecay;
use super::{ProdigyConfig, ProdigyKernel};
use crate::{
    dtypes::{Dtype, NotMixedPrecision},
    tensor::{Cpu, Error},
};

#[cfg(feature = "f16")]
use crate::dtypes::{f16, AMP};

#[cfg(feature = "f16")]
impl ProdigyKernel<AMP<f16>> for Cpu {
    fn prodigy_kernel(
        &self,
        k: i32,
        d: &mut f64,
        d_max: &mut f64,
        d_numerator: &mut f64,
        cfg: &ProdigyConfig,
        param: &mut Self::Vec,
        s: &mut Self::Vec,
        p0: &mut Self::Vec,
        p0b: &mut Self::Vec,
        moment1: &mut Self::Vec,
        moment2: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Error> {
        let mut d_denom_: f32 = 0.;
        let [beta1, beta2] = cfg.betas.map(|x| x as f32);
        let beta3 = cfg.beta3.unwrap_or_else(|| cfg.betas[1].sqrt()) as f32;

        let bias_correction = if cfg.use_bias_correction {
            // note: in here the first k = 1, whereas on the reference python code it's 0
            (1. - beta2.powi(k)).sqrt() / (1. - beta1.powi(k))
        } else {
            1.
        };
        let mut d_ = *d as f32;
        let mut d_max_ = *d_max as f32;
        let mut d_numerator_ = *d_numerator as f32 * beta3;
        let d0 = cfg.d0 as f32;
        let lr = cfg.lr as f32;

        let dlr = d_ * lr * bias_correction;

        for ((((((p, g), s), p0), p0b), m), v) in param
            .iter_mut()
            .zip(grad.iter().cloned())
            .zip(s.iter_mut())
            .zip(p0.iter_mut())
            .zip(p0b.iter_mut())
            .zip(moment1.iter_mut())
            .zip(moment2.iter_mut())
        {
            let p_ = p.0.to_f32();
            let mut g_ = g.0.to_f32();
            let mut s_ = s.0.to_f32();
            let p0b_ = p0b.0.to_f32();
            let mut m_ = m.0.to_f32();
            let mut v_ = v.0.to_f32();

            // initialize p0 if needed
            if p0b_ == 0. {
                p0b.0 = f16::from_f32(1.);
                *p0 = *p;
            }
            let p0_ = p0.0.to_f32();

            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                g_ += wd as f32 * p_;
            }

            if lr > 0. {
                d_numerator_ += (d_ / d0) * dlr * (g_ * (p0_ - p_));

                m_ = m_ * beta1 + d_ * g_ * (1. - beta1);
                v_ = v_ * beta2 + d_ * d_ * g_ * g_ * (1. - beta2);
                m.0 = f16::from_f32(m_);
                v.0 = f16::from_f32(v_);

                if cfg.safeguard_warmup {
                    s_ = s_ * beta3 + g_ * d_.powi(2) / d0;
                } else {
                    s_ = s_ * beta3 + g_ * d_ * dlr / d0;
                }
                s.0 = f16::from_f32(s_);

                d_denom_ += s_.abs();
            }
        }

        if d_denom_ == 0. {
            return Ok(());
        }

        let global_d_numerator = d_numerator_;
        let global_d_denom = d_denom_;
        if lr > 0. {
            let d_coef = cfg.d_coef as f32;
            let d_hat_ = d_coef * global_d_numerator / global_d_denom;
            if d_ == d0 {
                d_ = d_.max(d_hat_);
            }
            d_max_ = d_max_.max(d_hat_);
            let growth_rate = cfg.growth_rate as f32;
            d_ = d_max_.min(d_ * growth_rate);
        }

        *d = d_ as f64;
        *d_max = d_max_ as f64;
        *d_numerator = global_d_numerator as f64;

        let eps = cfg.eps as f32;

        for (p, (m, v)) in param
            .iter_mut()
            .zip(moment1.iter_mut().zip(moment2.iter_mut()))
        {
            let mut p_ = p.0.to_f32();
            let m_ = m.0.to_f32();
            let v_ = v.0.to_f32();

            let denom = v_.sqrt() + d_ * eps;

            if let Some(WeightDecay::Decoupled(wd)) = cfg.weight_decay {
                p_ *= 1. - wd as f32 * dlr;
            }

            p_ -= dlr * m_ / denom;
            p.0 -= f16::from_f32(p_);
        }

        Ok(())
    }
}

impl<E: num_traits::Float + Dtype + NotMixedPrecision> ProdigyKernel<E> for Cpu {
    fn prodigy_kernel(
        &self,
        k: i32,
        d: &mut f64,
        d_max: &mut f64,
        d_numerator: &mut f64,
        cfg: &ProdigyConfig,
        param: &mut Self::Vec,
        s: &mut Self::Vec,
        p0: &mut Self::Vec,
        p0b: &mut Self::Vec,
        moment1: &mut Self::Vec,
        moment2: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Error> {
        let mut d_denom_: E = E::zero();
        let [beta1, beta2] = cfg.betas.map(E::from_f64).map(Option::unwrap);

        #[allow(unused_imports)]
        let beta3 = E::from_f64(cfg.beta3.unwrap_or_else(|| {
            #[cfg(feature = "no-std")]
            use num_traits::Float;

            cfg.betas[1].sqrt()
        }))
        .unwrap();

        let bias_correction = if cfg.use_bias_correction {
            // note: in here the first k = 1, whereas on the reference python code it's 0
            (E::one() - beta2.powi(k)).sqrt() / (E::one() - beta1.powi(k))
        } else {
            E::one()
        };
        let mut d_ = E::from_f64(*d).unwrap();
        let mut d_max_ = E::from_f64(*d_max).unwrap();
        let mut d_numerator_ = E::from_f64(*d_numerator).unwrap() * beta3;
        let d0 = E::from_f64(cfg.d0).unwrap();
        let lr = E::from_f64(cfg.lr).unwrap();

        let dlr = d_ * lr * bias_correction;

        for ((((((p, mut g), s), p0), p0b), m), v) in param
            .iter_mut()
            .zip(grad.iter().cloned())
            .zip(s.iter_mut())
            .zip(p0.iter_mut())
            .zip(p0b.iter_mut())
            .zip(moment1.iter_mut())
            .zip(moment2.iter_mut())
        {
            // initialize p0 if needed
            if *p0b == E::zero() {
                *p0b = E::one();
                *p0 = *p;
            }

            if let Some(WeightDecay::L2(wd)) = cfg.weight_decay {
                g += E::from_f64(wd).unwrap() * *p;
            }

            if lr > E::zero() {
                d_numerator_ += (d_ / d0) * dlr * (g * (*p0 - *p));

                *m = *m * beta1 + d_ * g * (E::one() - beta1);
                *v = *v * beta2 + d_ * d_ * g * g * (E::one() - beta2);

                if cfg.safeguard_warmup {
                    *s = *s * beta3 + g * d_.powi(2) / d0
                } else {
                    *s = *s * beta3 + g * d_ * dlr / d0
                }

                d_denom_ += s.abs();
            }
        }

        if d_denom_ == E::zero() {
            return Ok(());
        }

        let global_d_numerator = d_numerator_;
        let global_d_denom = d_denom_;
        if lr > E::zero() {
            let d_coef = E::from_f64(cfg.d_coef).unwrap();
            let d_hat_ = d_coef * global_d_numerator / global_d_denom;
            if d_ == d0 {
                d_ = d_.max(d_hat_);
            }
            d_max_ = d_max_.max(d_hat_);
            let growth_rate = E::from_f64(cfg.growth_rate).unwrap();
            d_ = d_max_.min(d_ * growth_rate);
        }

        *d = d_.to_f64().unwrap();
        *d_max = d_max_.to_f64().unwrap();
        *d_numerator = global_d_numerator.to_f64().unwrap();

        let eps = E::from_f64(cfg.eps).unwrap();

        for (p, (m, v)) in param
            .iter_mut()
            .zip(moment1.iter_mut().zip(moment2.iter_mut()))
        {
            let denom = v.sqrt() + d_ * eps;

            if let Some(WeightDecay::Decoupled(wd)) = cfg.weight_decay {
                *p *= E::one() - E::from_f64(wd).unwrap() * dlr;
            }

            *p -= dlr * *m / denom;
        }

        Ok(())
    }
}
