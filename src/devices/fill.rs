use super::{AllocateZeros, Cpu};
use crate::arrays::CountElements;

/// Fills all elements with the specified function
pub trait FillElements<T: CountElements>: Sized + AllocateZeros {
    fn fill<F: FnMut() -> T::Dtype>(out: &mut T, f: &mut F);

    fn filled<F: FnMut() -> T::Dtype>(f: &mut F) -> Box<T> {
        let mut out = Self::zeros();
        Self::fill(&mut out, f);
        out
    }
}

impl FillElements<f32> for Cpu {
    fn fill<F: FnMut() -> f32>(out: &mut f32, f: &mut F) {
        *out = f();
    }
}

impl<T: CountElements, const M: usize> FillElements<[T; M]> for Cpu
where
    Self: FillElements<T>,
{
    fn fill<F: FnMut() -> T::Dtype>(out: &mut [T; M], f: &mut F) {
        for i in 0..M {
            Self::fill(&mut out[i], f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::ZeroElements;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_fill_rng() {
        let mut rng = thread_rng();
        let mut t: [f32; 5] = ZeroElements::ZEROS;
        Cpu::fill(&mut t, &mut || rng.gen_range(0.0..1.0));
        for i in 0..5 {
            assert!(t[i] < 1.0 && 0.0 <= t[i]);
        }
    }

    #[test]
    fn test_0d_fill() {
        let mut t: f32 = ZeroElements::ZEROS;
        Cpu::fill(&mut t, &mut || 1.0);
        assert_eq!(t, 1.0);
        Cpu::fill(&mut t, &mut || 2.0);
        assert_eq!(t, 2.0);
    }

    #[test]
    fn test_1d_fill() {
        let mut t: [f32; 5] = ZeroElements::ZEROS;
        Cpu::fill(&mut t, &mut || 1.0);
        assert_eq!(t, [1.0; 5]);
        Cpu::fill(&mut t, &mut || 2.0);
        assert_eq!(t, [2.0; 5]);
    }

    #[test]
    fn test_2d_fill() {
        let mut t: [[f32; 3]; 5] = ZeroElements::ZEROS;
        Cpu::fill(&mut t, &mut || 1.0);
        assert_eq!(t, [[1.0; 3]; 5]);
        Cpu::fill(&mut t, &mut || 2.0);
        assert_eq!(t, [[2.0; 3]; 5]);
    }

    #[test]
    fn test_3d_fill() {
        let mut t: [[[f32; 2]; 3]; 5] = ZeroElements::ZEROS;
        Cpu::fill(&mut t, &mut || 1.0);
        assert_eq!(t, [[[1.0; 2]; 3]; 5]);
        Cpu::fill(&mut t, &mut || 2.0);
        assert_eq!(t, [[[2.0; 2]; 3]; 5]);
    }
}
