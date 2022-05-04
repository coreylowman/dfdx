pub trait ZeroElements {
    const ZEROS: Self;
}

impl ZeroElements for f32 {
    const ZEROS: Self = 0.0;
}

impl<T: ZeroElements, const M: usize> ZeroElements for [T; M] {
    const ZEROS: Self = [T::ZEROS; M];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0d_zeros() {
        let t: f32 = ZeroElements::ZEROS;
        assert_eq!(t, 0.0);
    }

    #[test]
    fn test_1d_zeros() {
        let t: [f32; 5] = ZeroElements::ZEROS;
        assert_eq!(t, [0.0; 5]);
    }

    #[test]
    fn test_2d_zeros() {
        let t: [[f32; 3]; 5] = ZeroElements::ZEROS;
        assert_eq!(t, [[0.0; 3]; 5]);
    }

    #[test]
    fn test_3d_zeros() {
        let t: [[[f32; 2]; 3]; 5] = ZeroElements::ZEROS;
        assert_eq!(t, [[[0.0; 2]; 3]; 5]);
    }
}
