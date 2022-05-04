use super::CountElements;

pub trait ReduceElements: CountElements {
    fn reduce<F: FnMut(f32, f32) -> f32 + Copy>(&self, f: F) -> f32;

    fn sum(&self) -> f32 {
        self.reduce(|a, b| a + b)
    }

    fn max(&self) -> f32 {
        self.reduce(f32::max)
    }

    fn min(&self) -> f32 {
        self.reduce(f32::min)
    }

    fn mean(&self) -> f32 {
        self.sum() / Self::NUM_ELEMENTS as f32
    }
}

impl ReduceElements for f32 {
    fn reduce<F: FnMut(f32, f32) -> f32 + Copy>(&self, _: F) -> f32 {
        *self
    }
}

impl<T: ReduceElements, const M: usize> ReduceElements for [T; M] {
    fn reduce<F: FnMut(f32, f32) -> f32 + Copy>(&self, f: F) -> f32 {
        (0..M).map(|i| self[i].reduce(f)).reduce(f).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_0d() {
        assert_eq!(0.0.reduce(|a, b| a + b), 0.0);
        assert_eq!(0.0.sum(), 0.0);
        assert_eq!(0.0.mean(), 0.0);
        assert_eq!(0.0.max(), 0.0);
        assert_eq!(0.0.min(), 0.0);
    }

    #[test]
    fn test_reduce_1d() {
        let t = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(t.reduce(|a, b| a * b), 24.0);
        assert_eq!(t.sum(), 10.0);
        assert_eq!(t.mean(), 2.5);
        assert_eq!(t.max(), 4.0);
        assert_eq!(t.min(), 1.0);
    }

    #[test]
    fn test_reduce_2d() {
        let t = [[1.0, 2.0, 3.0, 4.0], [5.0, -1.0, 3.14, 0.0]];
        assert_eq!(t.reduce(|a, b| a * b), 0.0);
        assert_eq!(t.sum(), 17.14);
        assert_eq!(t.mean(), 2.1425);
        assert_eq!(t.max(), 5.0);
        assert_eq!(t.min(), -1.0);
    }

    #[test]
    fn test_reduce_3d() {
        let t = [[[1.0, 2.0], [2.0, 3.0]], [[1.0, 0.5], [0.5, 1.0 / 3.0]]];
        assert_eq!(t.reduce(|a, b| a * b), 1.0);
        assert!((t.sum() - (10.0 + 1.0 / 3.0)).abs() < 1e-6);
        assert_eq!(t.mean(), t.sum() / 8.0);
        assert_eq!(t.max(), 3.0);
        assert_eq!(t.min(), 1.0 / 3.0);
    }
}
