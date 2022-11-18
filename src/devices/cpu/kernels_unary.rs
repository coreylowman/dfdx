use super::Cpu;
use crate::arrays::Shape;
use crate::devices::{device::*, unary_ops};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Standard;
use std::sync::Arc;

trait Derivatives<E> {
    fn f(&self, x: &E) -> E;
    fn df(&self, x: &E) -> E;
}

impl Derivatives<f32> for unary_ops::Negate {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        -x
    }
    #[inline(always)]
    fn df(&self, _: &f32) -> f32 {
        -1.0
    }
}

impl Derivatives<f32> for unary_ops::ReLU {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.max(0.0)
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        if x > &0.0 {
            1.0
        } else {
            0.0
        }
    }
}

impl Derivatives<f32> for unary_ops::Square {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.powi(2)
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        2.0 * x
    }
}

impl Derivatives<f32> for unary_ops::Sqrt {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.sqrt()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        0.5 / x.sqrt()
    }
}

impl Derivatives<f32> for unary_ops::Tanh {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.tanh()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        1.0 - x.tanh().powi(2)
    }
}

impl Derivatives<f32> for unary_ops::Sigmoid {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        let fx = 1.0 / (1.0 + (-x).exp());
        fx * (1.0 - fx)
    }
}

impl Derivatives<f32> for unary_ops::Sin {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.sin()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        x.cos()
    }
}

impl Derivatives<f32> for unary_ops::Cos {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.cos()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        -x.sin()
    }
}

impl Derivatives<f32> for unary_ops::Exp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.exp()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        x.exp()
    }
}

impl Derivatives<f32> for unary_ops::Ln {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.ln()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        1.0 / x
    }
}

impl Derivatives<f32> for unary_ops::Abs {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.abs()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        if x == &0.0 {
            0.0
        } else {
            x.signum()
        }
    }
}

impl Derivatives<f32> for unary_ops::Clamp<f32> {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.clamp(self.min, self.max)
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        if (self.min..=self.max).contains(x) {
            1.0
        } else {
            0.0
        }
    }
}

impl Derivatives<f32> for unary_ops::Powi {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.powi(self.0)
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        self.0 as f32 * x.powi(self.0 - 1)
    }
}

impl Derivatives<f32> for unary_ops::Pow<f32> {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.powf(self.0)
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        self.0 * x.powf(self.0 - 1.0)
    }
}

impl Derivatives<f32> for unary_ops::NansTo<f32> {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        if x.is_nan() {
            self.0
        } else {
            *x
        }
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        if x.is_nan() {
            0.0
        } else {
            1.0
        }
    }
}

impl Derivatives<f32> for unary_ops::ScalarAdd<f32> {
    fn f(&self, x: &f32) -> f32 {
        x + self.0
    }
    fn df(&self, _: &f32) -> f32 {
        1.0
    }
}

impl Derivatives<f32> for unary_ops::ScalarSub<f32> {
    fn f(&self, x: &f32) -> f32 {
        x - self.0
    }
    fn df(&self, _: &f32) -> f32 {
        1.0
    }
}

impl Derivatives<f32> for unary_ops::ScalarMul<f32> {
    fn f(&self, x: &f32) -> f32 {
        x * self.0
    }
    fn df(&self, _: &f32) -> f32 {
        self.0
    }
}

impl Derivatives<f32> for unary_ops::ScalarDiv<f32> {
    fn f(&self, x: &f32) -> f32 {
        x / self.0
    }
    fn df(&self, _: &f32) -> f32 {
        1.0 / self.0
    }
}

impl<Op: Derivatives<f32>, S: Shape> UnaryKernel<Op, S, S, f32> for Cpu {
    fn unary_fwd(
        &self,
        op: Op,
        inp: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, f32>, Self::Err> {
        let mut out: Self::Storage<S, f32> = inp.try_clone()?;
        for x in Arc::make_mut(&mut out.data).iter_mut() {
            *x = op.f(x);
        }
        Ok(out)
    }

    fn unary_bwd(
        &self,
        op: Op,
        inp: &Self::Storage<S, f32>,
        grad_inp: &mut Self::Storage<S, f32>,
        grad_out: &Self::Storage<S, f32>,
    ) {
        assert_eq!(grad_inp.data.len(), grad_out.data.len());
        assert_eq!(inp.data.len(), grad_out.data.len());
        let data = Arc::make_mut(&mut grad_inp.data);
        for (i, data_i) in data.iter_mut().enumerate() {
            *data_i += op.df(&inp.data[i]) * grad_out.data[i];
        }
    }
}

impl<S: Shape> UnaryKernel<unary_ops::Dropout, S, S, f32> for Cpu {
    fn unary_fwd(
        &self,
        op: unary_ops::Dropout,
        inp: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, f32>, Self::Err> {
        let mut rng = StdRng::seed_from_u64(op.seed);
        let mut out: Self::Storage<S, f32> = inp.try_clone()?;
        let data = Arc::make_mut(&mut out.data);
        for x in data.iter_mut() {
            let val: f32 = rng.sample(Standard);
            *x = if val < op.prob {
                0.0
            } else {
                *x / (1.0 - op.prob)
            };
        }
        Ok(out)
    }

    fn unary_bwd(
        &self,
        op: unary_ops::Dropout,
        inp: &Self::Storage<S, f32>,
        grad_inp: &mut Self::Storage<S, f32>,
        grad_out: &Self::Storage<S, f32>,
    ) {
        let mut rng = StdRng::seed_from_u64(op.seed);
        assert_eq!(grad_inp.data.len(), grad_out.data.len());
        assert_eq!(inp.data.len(), grad_out.data.len());
        let data = Arc::make_mut(&mut grad_inp.data);
        for (i, data_i) in data.iter_mut().enumerate() {
            let val: f32 = rng.sample(Standard);
            *data_i += if val < op.prob {
                0.0
            } else {
                1.0 / (1.0 - op.prob)
            } * grad_out.data[i];
        }
    }
}

// impl<S: Shape, E: Dtype> UnaryKernel<unary_ops::Select<Rank0, Self>, S, todo!(), E> for Cpu {}
