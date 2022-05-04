pub trait MapInnerElements {
    type Inner;
    fn map_assign_inner<F: FnMut(&mut Self::Inner) + Copy>(&mut self, f: F);
}

impl<const M: usize> MapInnerElements for [f32; M] {
    type Inner = Self;
    fn map_assign_inner<F: FnMut(&mut Self::Inner) + Copy>(&mut self, mut f: F) {
        f(self)
    }
}

impl<T: MapInnerElements, const M: usize> MapInnerElements for [T; M] {
    type Inner = T::Inner;
    fn map_assign_inner<F: FnMut(&mut Self::Inner) + Copy>(&mut self, f: F) {
        for i in 0..M {
            self[i].map_assign_inner(f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_map_inner() {
        let mut t = [0.0; 5];
        t.map_assign_inner(|f| {
            f[0] = 1.0;
            f[4] = 2.0;
        });
        assert_eq!(t, [1.0, 0.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_2d_map_inner() {
        let mut t = [[0.0; 3], [1.0; 3], [2.0; 3], [3.0; 3], [4.0; 3]];
        t.map_assign_inner(|f| *f = [f.iter().sum(); 3]);
        assert_eq!(t, [[0.0; 3], [3.0; 3], [6.0; 3], [9.0; 3], [12.0; 3]])
    }

    #[test]
    fn test_3d_map_inner() {
        let mut t = [[[1.0; 2]; 3]; 5];
        t.map_assign_inner(|f| *f = [2.0; 2]);
        assert_eq!(t, [[[2.0; 2]; 3]; 5]);
    }
}
