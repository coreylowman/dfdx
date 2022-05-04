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
    fn test_reduce() {
        todo!("");
    }
}
