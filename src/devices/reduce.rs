use num_traits::{Float, FromPrimitive, Num};

use super::Cpu;
use crate::arrays::CountElements;

/// Reduce an entire Nd array to 1 value
pub trait ReduceElements<T: CountElements> {
    fn reduce<F: FnMut(T::Dtype, T::Dtype) -> T::Dtype>(inp: &T, f: &mut F) -> T::Dtype;

    fn sum(inp: &T) -> T::Dtype
    where
        T::Dtype: Num,
    {
        Self::reduce(inp, &mut |a, b| a + b)
    }

    fn mean(inp: &T) -> T::Dtype
    where
        T::Dtype: Float + FromPrimitive,
    {
        Self::sum(inp) / T::Dtype::from_usize(T::NUM_ELEMENTS).unwrap()
    }

    fn max(inp: &T) -> T::Dtype
    where
        T::Dtype: Float,
    {
        Self::reduce(inp, &mut T::Dtype::max)
    }

    fn min(inp: &T) -> T::Dtype
    where
        T::Dtype: Float,
    {
        Self::reduce(inp, &mut T::Dtype::min)
    }
}

impl ReduceElements<f32> for Cpu {
    fn reduce<F: FnMut(f32, f32) -> f32>(inp: &f32, _f: &mut F) -> f32 {
        *inp
    }
}

impl<T: CountElements, const M: usize> ReduceElements<[T; M]> for Cpu
where
    Self: ReduceElements<T>,
{
    fn reduce<F: FnMut(T::Dtype, T::Dtype) -> T::Dtype>(inp: &[T; M], f: &mut F) -> T::Dtype {
        let mut result = None;
        for i in 0..M {
            let partial = Self::reduce(&inp[i], f);
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
    use super::super::*;
    use super::*;

    #[test]
    fn test_reduce_0d() {
        assert_eq!(Cpu::reduce(&0.0, &mut |a, b| a + b), 0.0);
        assert_eq!(Cpu::sum(&0.0), 0.0);
        assert_eq!(Cpu::mean(&0.0), 0.0);
        assert_eq!(Cpu::max(&0.0), 0.0);
        assert_eq!(Cpu::min(&0.0), 0.0);
    }

    #[test]
    fn test_reduce_1d() {
        let t = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(Cpu::reduce(&t, &mut |a, b| a * b), 24.0);
        assert_eq!(Cpu::sum(&t), 10.0);
        assert_eq!(Cpu::mean(&t), 2.5);
        assert_eq!(Cpu::max(&t), 4.0);
        assert_eq!(Cpu::min(&t), 1.0);
    }

    #[test]
    fn test_reduce_2d() {
        let t = [[1.0, 2.0, 3.0, 4.0], [5.0, -1.0, 3.14, 0.0]];
        assert_eq!(Cpu::reduce(&t, &mut |a, b| a * b), 0.0);
        assert_eq!(Cpu::sum(&t), 17.14);
        assert_eq!(Cpu::mean(&t), 2.1425);
        assert_eq!(Cpu::max(&t), 5.0);
        assert_eq!(Cpu::min(&t), -1.0);
    }

    #[test]
    fn test_reduce_3d() {
        let t = [[[1.0, 2.0], [2.0, 3.0]], [[1.0, 0.5], [0.5, 1.0 / 3.0]]];
        assert_eq!(Cpu::reduce(&t, &mut |a, b| a * b), 1.0);
        let sum = Cpu::sum(&t);
        assert!((sum - (10.0 + 1.0 / 3.0)).abs() < 1e-6);
        assert_eq!(Cpu::mean(&t), sum / 8.0);
        assert_eq!(Cpu::max(&t), 3.0);
        assert_eq!(Cpu::min(&t), 1.0 / 3.0);
    }
}
