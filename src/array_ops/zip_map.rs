use super::ZeroElements;

pub trait ZipMapElements<Rhs> {
    fn zip_map<F: FnMut(&f32, &f32) -> f32 + Copy>(&self, other: &Rhs, f: F) -> Self;
    fn zip_map_assign<F: FnMut(&mut f32, &f32) + Copy>(&mut self, other: &Rhs, f: F);
}

impl ZipMapElements<f32> for f32 {
    fn zip_map<F: FnMut(&f32, &f32) -> f32 + Copy>(&self, other: &Self, mut f: F) -> Self {
        f(self, other)
    }

    fn zip_map_assign<F: FnMut(&mut f32, &f32) + Copy>(&mut self, other: &Self, mut f: F) {
        f(self, other)
    }
}

impl<const M: usize> ZipMapElements<f32> for [f32; M] {
    fn zip_map<F: FnMut(&f32, &f32) -> f32 + Copy>(&self, other: &f32, f: F) -> Self {
        let mut result = Self::ZEROS;
        for i in 0..M {
            result[i] = self[i].zip_map(other, f);
        }
        result
    }
    fn zip_map_assign<F: FnMut(&mut f32, &f32) + Copy>(&mut self, other: &f32, f: F) {
        for i in 0..M {
            self[i].zip_map_assign(other, f);
        }
    }
}

impl<Rhs, T: ZipMapElements<Rhs> + ZeroElements, const M: usize> ZipMapElements<[Rhs; M]>
    for [T; M]
{
    fn zip_map<F: FnMut(&f32, &f32) -> f32 + Copy>(&self, other: &[Rhs; M], f: F) -> Self {
        let mut result = Self::ZEROS;
        for i in 0..M {
            result[i] = self[i].zip_map(&other[i], f);
        }
        result
    }

    fn zip_map_assign<F: FnMut(&mut f32, &f32) + Copy>(&mut self, other: &[Rhs; M], f: F) {
        for i in 0..M {
            self[i].zip_map_assign(&other[i], f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0d_zip_map() {
        let mut a = 1.0;
        let b = 2.0;
        assert_eq!(a.zip_map(&b, |x, y| x + y), 3.0);
        a.zip_map_assign(&b, |x, y| *x += y);
        assert_eq!(a, 3.0)
    }

    #[test]
    fn test_1d_zip_map() {
        let mut a = [1.0; 3];
        let b = [2.0; 3];
        assert_eq!(a.zip_map(&b, |x, y| x + y), [3.0; 3]);
        a.zip_map_assign(&b, |x, y| *x += y);
        assert_eq!(a, [3.0; 3]);
    }

    #[test]
    fn test_1d_zip_map_broadcast_inner() {
        let mut a = [1.0; 3];
        let b = 2.0;
        assert_eq!(a.zip_map(&b, |x, y| x + y), [3.0; 3]);
        a.zip_map_assign(&b, |x, y| *x += y);
        assert_eq!(a, [3.0; 3]);
    }

    #[test]
    fn test_2d_zip_map() {
        let mut a = [[2.0; 2]; 3];
        let b = [[3.0; 2]; 3];
        assert_eq!(a.zip_map(&b, |x, y| x * y), [[6.0; 2]; 3]);
        a.zip_map_assign(&b, |x, y| *x *= y);
        assert_eq!(a, [[6.0; 2]; 3]);
    }

    #[test]
    fn test_2d_zip_map_broadcast_inner() {
        let mut a = [[2.0; 2]; 3];
        let b = [3.0; 3];
        assert_eq!(a.zip_map(&b, |x, y| x * y), [[6.0; 2]; 3]);
        a.zip_map_assign(&b, |x, y| *x *= y);
        assert_eq!(a, [[6.0; 2]; 3]);
    }

    #[test]
    fn test_3d_zip_map() {
        let mut a = [[[2.0; 5]; 2]; 3];
        let b = [[[2.0; 5]; 2]; 3];
        assert_eq!(a.zip_map(&b, |x, y| x / y), [[[1.0; 5]; 2]; 3]);
        a.zip_map_assign(&b, |x, y| *x /= y);
        assert_eq!(a, [[[1.0; 5]; 2]; 3]);
    }

    #[test]
    fn test_3d_zip_map_broadcast_inner() {
        let mut a = [[[2.0; 5]; 2]; 3];
        let b = [[2.0; 2]; 3];
        assert_eq!(a.zip_map(&b, |x, y| x / y), [[[1.0; 5]; 2]; 3]);
        a.zip_map_assign(&b, |x, y| *x /= y);
        assert_eq!(a, [[[1.0; 5]; 2]; 3]);
    }
}
