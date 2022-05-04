pub trait CountElements {
    const NUM_ELEMENTS: usize;
}

impl CountElements for f32 {
    const NUM_ELEMENTS: usize = 1;
}

impl<T: CountElements, const M: usize> CountElements for [T; M] {
    const NUM_ELEMENTS: usize = M * T::NUM_ELEMENTS;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0d_count() {
        assert_eq!(1, f32::NUM_ELEMENTS);
    }

    #[test]
    fn test_1d_count() {
        assert_eq!(5, <[f32; 5]>::NUM_ELEMENTS);
    }

    #[test]
    fn test_2d_count() {
        assert_eq!(15, <[[f32; 3]; 5]>::NUM_ELEMENTS);
    }

    #[test]
    fn test_3d_count() {
        assert_eq!(30, <[[[f32; 2]; 3]; 5]>::NUM_ELEMENTS);
    }
}
