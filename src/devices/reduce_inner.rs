use super::{AllocateZeros, Cpu};
use crate::arrays::CountElements;

pub trait ReduceInnerElements<T>: Sized + AllocateZeros {
    type Output: Sized + CountElements;

    fn reduce_inner_into<F: FnMut(f32, f32) -> f32 + Copy>(inp: &T, out: &mut Self::Output, f: F);

    fn reduce_inner<F: FnMut(f32, f32) -> f32 + Copy>(inp: &T, f: F) -> Box<Self::Output> {
        let mut out = Self::zeros();
        Self::reduce_inner_into(inp, &mut out, f);
        out
    }
}

impl<const M: usize> ReduceInnerElements<[f32; M]> for Cpu {
    type Output = f32;
    fn reduce_inner_into<F: FnMut(f32, f32) -> f32 + Copy>(
        inp: &[f32; M],
        out: &mut Self::Output,
        f: F,
    ) {
        *out = inp.iter().cloned().reduce(f).unwrap();
    }
}

impl<T, const M: usize> ReduceInnerElements<[T; M]> for Cpu
where
    Cpu: ReduceInnerElements<T>,
{
    type Output = [<Self as ReduceInnerElements<T>>::Output; M];
    fn reduce_inner_into<F: FnMut(f32, f32) -> f32 + Copy>(
        inp: &[T; M],
        out: &mut Self::Output,
        f: F,
    ) {
        for i in 0..M {
            Self::reduce_inner_into(&inp[i], &mut out[i], f);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::ZeroElements;

    use super::*;

    #[test]
    fn test_reduce_inner_1d() {
        let t = [1.0, 2.0, 3.0];
        let mut out = ZeroElements::ZEROS;
        Cpu::reduce_inner_into(&t, &mut out, f32::max);
        assert_eq!(out, 3.0);
    }

    #[test]
    fn test_reduce_inner_2d() {
        let t = [[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]];
        let mut out = ZeroElements::ZEROS;
        Cpu::reduce_inner_into(&t, &mut out, f32::max);
        assert_eq!(out, [3.0, 6.0]);
    }

    #[test]
    fn test_reduce_inner_3d() {
        let t = [
            [[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]],
            [[-1.0, -2.0, -3.0], [-6.0, -5.0, -4.0]],
        ];
        let mut out = ZeroElements::ZEROS;
        Cpu::reduce_inner_into(&t, &mut out, f32::max);
        assert_eq!(out, [[3.0, 6.0], [-1.0, -4.0]]);
    }
}
