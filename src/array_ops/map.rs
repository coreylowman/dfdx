use super::ZeroElements;

pub trait MapElements: Sized {
    fn map_elems<F: FnMut(&f32) -> f32 + Copy>(&self, f: F) -> Self;
    fn mapv_elems<F: FnMut(f32) -> f32 + Copy>(&self, f: F) -> Self;
    fn map_assign_elems<F: FnMut(&mut f32) + Copy>(&mut self, f: F);

    fn scale(&self, s: f32) -> Self {
        self.map_elems(|v| v * s)
    }
}

impl MapElements for f32 {
    fn map_elems<F: FnMut(&f32) -> f32 + Copy>(&self, mut f: F) -> Self {
        f(self)
    }

    fn mapv_elems<F: FnMut(f32) -> f32 + Copy>(&self, mut f: F) -> Self {
        f(*self)
    }

    fn map_assign_elems<F: FnMut(&mut f32) + Copy>(&mut self, mut f: F) {
        f(self)
    }
}

impl<T: MapElements + ZeroElements, const M: usize> MapElements for [T; M] {
    fn map_elems<F: FnMut(&f32) -> f32 + Copy>(&self, f: F) -> Self {
        let mut result = Self::ZEROS;
        for i in 0..M {
            result[i] = self[i].map_elems(f);
        }
        result
    }

    fn mapv_elems<F: FnMut(f32) -> f32 + Copy>(&self, f: F) -> Self {
        let mut result = Self::ZEROS;
        for i in 0..M {
            result[i] = self[i].mapv_elems(f);
        }
        result
    }

    fn map_assign_elems<F: FnMut(&mut f32) + Copy>(&mut self, f: F) {
        for i in 0..M {
            self[i].map_assign_elems(f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0d_map() {
        let mut t = 1.0;
        let expected = 2.0;
        assert_eq!(t.map_elems(|v| v * 2.0), expected);
        assert_eq!(t.mapv_elems(|v| v * 2.0), expected);
        t.map_assign_elems(|v| *v *= 2.0);
        assert_eq!(t, expected);
    }

    #[test]
    fn test_1d_map() {
        let mut t = [0.0, 1.0, 2.0];
        let expected = [0.0, 2.0, 4.0];
        assert_eq!(t.map_elems(|v| v * 2.0), expected);
        assert_eq!(t.mapv_elems(|v| v * 2.0), expected);
        t.map_assign_elems(|v| *v *= 2.0);
        assert_eq!(t, expected);
    }

    #[test]
    fn test_2d_map() {
        let mut t = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
        let expected = [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]];
        assert_eq!(t.map_elems(|v| v * 2.0), expected);
        assert_eq!(t.mapv_elems(|v| v * 2.0), expected);
        t.map_assign_elems(|v| *v *= 2.0);
        assert_eq!(t, expected);
    }

    #[test]
    fn test_3d_map() {
        let mut t = [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ];
        let expected = [
            [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]],
            [[12.0, 14.0, 16.0], [18.0, 20.0, 22.0]],
        ];
        assert_eq!(t.map_elems(|v| v * 2.0), expected);
        assert_eq!(t.mapv_elems(|v| v * 2.0), expected);
        t.map_assign_elems(|v| *v *= 2.0);
        assert_eq!(t, expected);
    }
}
