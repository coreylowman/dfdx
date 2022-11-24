use crate::arrays::Shape;
use crate::devices::cpu::Cpu;
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

