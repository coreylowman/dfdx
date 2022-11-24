use super::HuberErrorKernelOp;
use crate::tensor_ops::utils::cpu::BinaryDerivative;

impl BinaryDerivative<f32> for HuberErrorKernelOp<f32> {
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
