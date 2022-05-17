use super::{AllocateZeros, Cpu};
use crate::arrays::CountElements;

pub trait MapElements<T>: Sized {
    fn map_into<F: FnMut(f32) -> f32 + Copy>(inp: &T, out: &mut T, f: F);
    fn map_assign<F: FnMut(&mut f32) + Copy>(inp: &mut T, f: F);
    fn map<F: FnMut(f32) -> f32 + Copy>(inp: &T, f: F) -> Box<T>
    where
        T: CountElements,
        Self: AllocateZeros<T>,
    {
        let mut out = Self::zeros();
        Self::map_into(inp, &mut out, f);
        out
    }

    fn scale(inp: &T, s: f32) -> Box<T>
    where
        T: CountElements,
        Self: AllocateZeros<T>,
    {
        Self::map(inp, |v| v * s)
    }
}

impl MapElements<f32> for Cpu {
    fn map_into<F: FnMut(f32) -> f32 + Copy>(inp: &f32, out: &mut f32, mut f: F) {
        *out = f(*inp);
    }
    fn map_assign<F: FnMut(&mut f32) + Copy>(inp: &mut f32, mut f: F) {
        f(inp);
    }
}

impl<T, const M: usize> MapElements<[T; M]> for Cpu
where
    Cpu: MapElements<T>,
{
    fn map_into<F: FnMut(f32) -> f32 + Copy>(inp: &[T; M], out: &mut [T; M], f: F) {
        for i in 0..M {
            Self::map_into(&inp[i], &mut out[i], f);
        }
    }
    fn map_assign<F: FnMut(&mut f32) + Copy>(inp: &mut [T; M], f: F) {
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
        Cpu::map_into(&t, &mut out, |v| v * 2.0);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_1d_map() {
        let t = [0.0, 1.0, 2.0];
        let expected = [0.0, 2.0, 4.0];
        let mut out = ZeroElements::ZEROS;
        Cpu::map_into(&t, &mut out, |v| v * 2.0);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_2d_map() {
        let t = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
        let expected = [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]];
        let mut out = ZeroElements::ZEROS;
        Cpu::map_into(&t, &mut out, |v| v * 2.0);
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
        Cpu::map_into(&t, &mut out, |v| v * 2.0);
        assert_eq!(out, expected);
    }
}
