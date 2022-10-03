use super::{AllocateZeros, Cpu};
use crate::arrays::CountElements;

/// Fills all elements with the specified function
pub trait FillElements<T: CountElements>: Sized + AllocateZeros {
    fn fill<F: FnMut(&mut T::Dtype)>(out: &mut T, f: &mut F);

    fn filled<F: FnMut(&mut T::Dtype)>(f: &mut F) -> Box<T> {
        let mut out = Self::zeros();
        Self::fill(&mut out, f);
        out
    }
}

impl FillElements<f32> for Cpu {
    fn fill<F: FnMut(&mut f32)>(out: &mut f32, f: &mut F) {
        f(out)
    }
}

impl<T: CountElements, const M: usize> FillElements<[T; M]> for Cpu
where
    Self: FillElements<T>,
{
    fn fill<F: FnMut(&mut T::Dtype)>(out: &mut [T; M], f: &mut F) {
        for out_i in out.iter_mut() {
            Self::fill(out_i, f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::ZeroElements;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_fill_rng() {
        let mut rng = thread_rng();
        let mut t: [f32; 5] = ZeroElements::ZEROS;
        Cpu::fill(&mut t, &mut |v| *v = rng.gen_range(0.0..1.0));
        for &item in t.iter() {
            assert!((0.0..1.0).contains(&item));
        }
    }

    #[test]
    fn test_0d_fill() {
        let mut t: f32 = ZeroElements::ZEROS;
        Cpu::fill(&mut t, &mut |v| *v = 1.0);
        assert_eq!(t, 1.0);
        Cpu::fill(&mut t, &mut |v| *v = 2.0);
        assert_eq!(t, 2.0);
    }

    #[test]
    fn test_1d_fill() {
        let mut t: [f32; 5] = ZeroElements::ZEROS;
        Cpu::fill(&mut t, &mut |v| *v = 1.0);
        assert_eq!(t, [1.0; 5]);
        Cpu::fill(&mut t, &mut |v| *v = 2.0);
        assert_eq!(t, [2.0; 5]);
    }

    #[test]
    fn test_2d_fill() {
        let mut t: [[f32; 3]; 5] = ZeroElements::ZEROS;
        Cpu::fill(&mut t, &mut |v| *v = 1.0);
        assert_eq!(t, [[1.0; 3]; 5]);
        Cpu::fill(&mut t, &mut |v| *v = 2.0);
        assert_eq!(t, [[2.0; 3]; 5]);
    }

    #[test]
    fn test_3d_fill() {
        let mut t: [[[f32; 2]; 3]; 5] = ZeroElements::ZEROS;
        Cpu::fill(&mut t, &mut |v| *v = 1.0);
        assert_eq!(t, [[[1.0; 2]; 3]; 5]);
        Cpu::fill(&mut t, &mut |v| *v = 2.0);
        assert_eq!(t, [[[2.0; 2]; 3]; 5]);
    }
}
