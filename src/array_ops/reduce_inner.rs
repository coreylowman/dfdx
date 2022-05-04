use super::ZeroElements;

pub trait ReduceInnerElements: Sized {
    type Output: Sized + ZeroElements;
    fn reduce_inner<F: FnMut(f32, f32) -> f32 + Copy>(&self, f: F) -> Self::Output;
}

impl<const M: usize> ReduceInnerElements for [f32; M] {
    type Output = f32;
    fn reduce_inner<F: FnMut(f32, f32) -> f32 + Copy>(&self, f: F) -> Self::Output {
        self.iter().cloned().reduce(f).unwrap()
    }
}

impl<T: ReduceInnerElements, const M: usize> ReduceInnerElements for [T; M] {
    type Output = [<T as ReduceInnerElements>::Output; M];
    fn reduce_inner<F: FnMut(f32, f32) -> f32 + Copy>(&self, f: F) -> Self::Output {
        let mut result = Self::Output::ZEROS;
        for i in 0..M {
            result[i] = self[i].reduce_inner(f);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_inner_1d() {
        let t = [1.0, 2.0, 3.0];
        assert_eq!(t.reduce_inner(f32::max), 3.0);
    }

    #[test]
    fn test_reduce_inner_2d() {
        let t = [[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]];
        assert_eq!(t.reduce_inner(f32::max), [3.0, 6.0]);
    }

    #[test]
    fn test_reduce_inner_3d() {
        let t = [
            [[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]],
            [[-1.0, -2.0, -3.0], [-6.0, -5.0, -4.0]],
        ];
        assert_eq!(t.reduce_inner(f32::max), [[3.0, 6.0], [-1.0, -4.0]]);
    }
}
