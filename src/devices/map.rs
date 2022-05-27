use super::{AllocateZeros, Cpu};
use crate::arrays::CountElements;

/// Apply a generic function to all elements of Nd arrays.
pub trait MapElements<T: CountElements>: Sized + AllocateZeros {
    /// Calls `f` on every element of `inp`, and stores the result in `out`.
    fn map_into<F: FnMut(&T::Dtype) -> T::Dtype>(inp: &T, out: &mut T, f: &mut F);

    /// Calls `f` on every element of `inp`, which mutates `inp`.
    fn map_assign<F: FnMut(&mut T::Dtype)>(inp: &mut T, f: &mut F);

    /// Allocates using [AllocateZeros] and then calls [MapElements::map_into()]
    fn map<F: FnMut(&T::Dtype) -> T::Dtype>(inp: &T, mut f: F) -> Box<T> {
        let mut out = Self::zeros();
        Self::map_into(inp, &mut out, &mut f);
        out
    }
}

impl MapElements<f32> for Cpu {
    fn map_into<F: FnMut(&f32) -> f32>(inp: &f32, out: &mut f32, f: &mut F) {
        *out = f(inp);
    }

    fn map_assign<F: FnMut(&mut f32)>(inp: &mut f32, f: &mut F) {
        f(inp);
    }
}

impl<T: CountElements, const M: usize> MapElements<[T; M]> for Cpu
where
    Self: MapElements<T>,
{
    fn map_into<F: FnMut(&T::Dtype) -> T::Dtype>(inp: &[T; M], out: &mut [T; M], f: &mut F) {
        for i in 0..M {
            Self::map_into(&inp[i], &mut out[i], f);
        }
    }
    fn map_assign<F: FnMut(&mut T::Dtype)>(inp: &mut [T; M], f: &mut F) {
        for i in 0..M {
            Self::map_assign(&mut inp[i], f);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::ZeroElements;

    use super::*;

    #[test]
    fn test_0d_map() {
        let t = 1.0;
        let mut out = 0.0;
        let expected = 2.0;
        Cpu::map_into(&t, &mut out, &mut |v| v * 2.0);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_1d_map() {
        let t = [0.0, 1.0, 2.0];
        let expected = [0.0, 2.0, 4.0];
        let mut out = ZeroElements::ZEROS;
        Cpu::map_into(&t, &mut out, &mut |v| v * 2.0);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_2d_map() {
        let t = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
        let expected = [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]];
        let mut out = ZeroElements::ZEROS;
        Cpu::map_into(&t, &mut out, &mut |v| v * 2.0);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_3d_map() {
        let t = [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ];
        let expected = [
            [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]],
            [[12.0, 14.0, 16.0], [18.0, 20.0, 22.0]],
        ];
        let mut out = ZeroElements::ZEROS;
        Cpu::map_into(&t, &mut out, &mut |v| v * 2.0);
        assert_eq!(out, expected);
    }
}
