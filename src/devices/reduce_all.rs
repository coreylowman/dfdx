use super::Cpu;
use crate::arrays::CountElements;

/// Reduce an entire Nd array to 1 value
pub trait ReduceAllElements<T: CountElements> {
    fn reduce_all<F: FnMut(T::Dtype, T::Dtype) -> T::Dtype>(inp: &T, f: &mut F) -> T::Dtype;
}

impl ReduceAllElements<f32> for Cpu {
    fn reduce_all<F: FnMut(f32, f32) -> f32>(inp: &f32, _f: &mut F) -> f32 {
        *inp
    }
}

impl<T: CountElements, const M: usize> ReduceAllElements<[T; M]> for Cpu
where
    Self: ReduceAllElements<T>,
{
    fn reduce_all<F: FnMut(T::Dtype, T::Dtype) -> T::Dtype>(inp: &[T; M], f: &mut F) -> T::Dtype {
        let mut result = None;
        for inp_i in inp.iter() {
            let partial = Self::reduce_all(inp_i, f);
            result = match result {
                Some(r) => Some(f(r, partial)),
                None => Some(partial),
            };
        }
        result.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_0d() {
        assert_eq!(Cpu::reduce_all(&0.0, &mut |a, b| a + b), 0.0);
    }

    #[test]
    fn test_reduce_1d() {
        let t = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(Cpu::reduce_all(&t, &mut |a, b| a * b), 24.0);
    }

    #[test]
    fn test_reduce_2d() {
        let t = [[1.0, 2.0, 3.0, 4.0], [5.0, -1.0, 2.0, 0.0]];
        assert_eq!(Cpu::reduce_all(&t, &mut |a, b| a + b), 16.0);
    }

    #[test]
    fn test_reduce_3d() {
        let t = [[[1.0, 2.0], [2.0, 3.0]], [[1.0, 0.5], [0.5, 1.0 / 3.0]]];
        assert_eq!(Cpu::reduce_all(&t, &mut |a, b| a * b), 1.0);
    }
}
