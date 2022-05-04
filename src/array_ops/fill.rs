use super::ZeroElements;

pub trait FillElements: Sized + ZeroElements {
    fn filled_with<F: FnMut() -> f32>(f: &mut F) -> Self {
        let mut result = Self::ZEROS;
        result.fill_with(f);
        result
    }
    fn fill_with<F: FnMut() -> f32>(&mut self, f: &mut F);
}

impl FillElements for f32 {
    fn fill_with<F: FnMut() -> f32>(&mut self, f: &mut F) {
        *self = f();
    }
}

impl<T: FillElements + ZeroElements, const M: usize> FillElements for [T; M] {
    fn fill_with<F: FnMut() -> f32>(&mut self, f: &mut F) {
        for i in 0..M {
            self[i].fill_with(f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_fill_rng() {
        let mut rng = thread_rng();
        let t: [f32; 5] = FillElements::filled_with(&mut || rng.gen_range(0.0..1.0));
        for i in 0..5 {
            assert!(t[i] < 1.0 && 0.0 <= t[i]);
        }
    }

    #[test]
    fn test_0d_fill() {
        let mut t: f32 = FillElements::filled_with(&mut || 1.0);
        assert_eq!(t, 1.0);
        t.fill_with(&mut || 2.0);
        assert_eq!(t, 2.0);
    }

    #[test]
    fn test_1d_fill() {
        let mut t: [f32; 5] = FillElements::filled_with(&mut || 1.0);
        assert_eq!(t, [1.0; 5]);
        t.fill_with(&mut || 2.0);
        assert_eq!(t, [2.0; 5]);
    }

    #[test]
    fn test_2d_fill() {
        let mut t: [[f32; 3]; 5] = FillElements::filled_with(&mut || 1.0);
        assert_eq!(t, [[1.0; 3]; 5]);
        t.fill_with(&mut || 2.0);
        assert_eq!(t, [[2.0; 3]; 5]);
    }

    #[test]
    fn test_3d_fill() {
        let mut t: [[[f32; 2]; 3]; 5] = FillElements::filled_with(&mut || 1.0);
        assert_eq!(t, [[[1.0; 2]; 3]; 5]);
        t.fill_with(&mut || 2.0);
        assert_eq!(t, [[[2.0; 2]; 3]; 5]);
    }
}
