use super::super::utils::cpu::UnaryDerivative;
use super::ClampKernelOp;

impl UnaryDerivative<f32> for ClampKernelOp<f32> {
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
