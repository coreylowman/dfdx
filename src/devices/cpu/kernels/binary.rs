use crate::arrays::*;
use crate::devices::cpu::{Cpu, LendingIterator};
use crate::devices::{binary_ops, device::*};

trait Derivatives<T> {
    fn f(&self, x: &T, y: &T) -> T;
    fn dfdx(&self, x: &T, y: &T) -> T;
    fn dfdy(&self, x: &T, y: &T) -> T;
}

impl Derivatives<f32> for binary_ops::Add {
    #[inline(always)]
    fn f(&self, x: &f32, y: &f32) -> f32 {
        x + y
    }
    #[inline(always)]
    fn dfdx(&self, _: &f32, _: &f32) -> f32 {
        1.0
    }
    #[inline(always)]
    fn dfdy(&self, _: &f32, _: &f32) -> f32 {
        1.0
    }
}

impl Derivatives<f32> for binary_ops::Sub {
    #[inline(always)]
    fn f(&self, x: &f32, y: &f32) -> f32 {
        x - y
    }
    #[inline(always)]
    fn dfdx(&self, _: &f32, _: &f32) -> f32 {
        1.0
    }
    #[inline(always)]
    fn dfdy(&self, _: &f32, _: &f32) -> f32 {
        -1.0
    }
}

impl Derivatives<f32> for binary_ops::Mul {
    #[inline(always)]
    fn f(&self, x: &f32, y: &f32) -> f32 {
        x * y
    }
    #[inline(always)]
    fn dfdx(&self, _x: &f32, y: &f32) -> f32 {
        *y
    }
    #[inline(always)]
    fn dfdy(&self, x: &f32, _y: &f32) -> f32 {
        *x
    }
}

impl Derivatives<f32> for binary_ops::Div {
    #[inline(always)]
    fn f(&self, x: &f32, y: &f32) -> f32 {
        x / y
    }
    #[inline(always)]
    fn dfdx(&self, _: &f32, y: &f32) -> f32 {
        1.0 / y
    }
    #[inline(always)]
    fn dfdy(&self, x: &f32, y: &f32) -> f32 {
        -x / y.powi(2)
    }
}

impl Derivatives<f32> for binary_ops::MaxBinary {
    #[inline(always)]
    fn f(&self, x: &f32, y: &f32) -> f32 {
        x.max(*y)
    }
    #[inline(always)]
    fn dfdx(&self, x: &f32, y: &f32) -> f32 {
        if x > y {
            1.0
        } else if x < y {
            0.0
        } else {
            0.5
        }
    }
    #[inline(always)]
    fn dfdy(&self, x: &f32, y: &f32) -> f32 {
        if y > x {
            1.0
        } else if y < x {
            0.0
        } else {
            0.5
        }
    }
}

impl Derivatives<f32> for binary_ops::MinBinary {
    #[inline(always)]
    fn f(&self, x: &f32, y: &f32) -> f32 {
        x.min(*y)
    }
    #[inline(always)]
    fn dfdx(&self, x: &f32, y: &f32) -> f32 {
        if x < y {
            1.0
        } else if x > y {
            0.0
        } else {
            0.5
        }
    }

    #[inline(always)]
    fn dfdy(&self, x: &f32, y: &f32) -> f32 {
        if y < x {
            1.0
        } else if y > x {
            0.0
        } else {
            0.5
        }
    }
}

impl Derivatives<f32> for binary_ops::HuberError<f32> {
    #[inline(always)]
    fn f(&self, x: &f32, y: &f32) -> f32 {
        if (x - y).abs() < self.delta {
            (x - y).powi(2) * 0.5
        } else {
            (x - y).abs() * self.delta - 0.5 * self.delta * self.delta
        }
    }

    #[inline(always)]
    fn dfdx(&self, x: &f32, y: &f32) -> f32 {
        if (x - y) == 0.0 {
            0.0
        } else if (x - y).abs() < self.delta {
            x - y
        } else {
            (x - y).signum() * self.delta
        }
    }

    #[inline(always)]
    fn dfdy(&self, x: &f32, y: &f32) -> f32 {
        if (x - y) == 0.0 {
            0.0
        } else if (x - y).abs() < self.delta {
            y - x
        } else {
            (y - x).signum() * self.delta
        }
    }
}

impl Derivatives<f32> for binary_ops::BCEWithLogits {
    #[inline(always)]
    fn f(&self, logit: &f32, prob: &f32) -> f32 {
        logit.max(0.0) - logit * prob + (1.0 + (-logit.abs()).exp()).ln()
    }
    #[inline(always)]
    fn dfdx(&self, logit: &f32, prob: &f32) -> f32 {
        1.0 - prob - (1.0 + logit.exp()).recip()
    }

    #[inline(always)]
    fn dfdy(&self, logit: &f32, _: &f32) -> f32 {
        -logit
    }
}

impl<Op: Derivatives<f32>, const N: usize, S: Shape<Concrete = [usize; N]>>
    BinaryKernel<Op, S, S, S, f32> for Cpu
{
    fn binary_fwd(
        &self,
        op: Op,
        lhs: &Self::Storage<S, f32>,
        rhs: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, f32>, Self::Err> {
        let mut out: Self::Storage<S, f32> = self.try_zeros_like(lhs.shape)?;
        let mut lhs_iter = lhs.iter();
        let mut rhs_iter = rhs.iter();
        let mut out_iter = out.iter_mut();
        while let Some((o, (l, r))) = out_iter.next().zip(lhs_iter.next().zip(rhs_iter.next())) {
            *o = op.f(l, r);
        }
        Ok(out)
    }
    fn binary_bwd(
        &self,
        op: Op,
        lhs: &Self::Storage<S, f32>,
        grad_lhs: &mut Self::Storage<S, f32>,
        rhs: &Self::Storage<S, f32>,
        grad_rhs: &mut Self::Storage<S, f32>,
        grad_out: &Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        let mut lhs_iter = lhs.iter();
        let mut rhs_iter = rhs.iter();
        let mut grad_lhs_iter = grad_lhs.iter_mut();
        let mut grad_rhs_iter = grad_rhs.iter_mut();
        let mut grad_out_iter = grad_out.iter();
        for _ in 0..lhs.shape.num_elements() {
            let l = lhs_iter.next().unwrap();
            let r = rhs_iter.next().unwrap();
            let go = grad_out_iter.next().unwrap();
            let gl = grad_lhs_iter.next().unwrap();
            *gl += op.dfdx(l, r) * go;
            let gr = grad_rhs_iter.next().unwrap();
            *gr += op.dfdy(l, r) * go;
        }
        Ok(())
    }
}
