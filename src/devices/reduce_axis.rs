//! Reduction of any single axis.
//!
//! # Implementation details
//!
//! To reduce anything past the first dimension, you can recurse
//! one level lower and reduce the axis you are reducing by 1.
//! For example when reducing a 2d axis 1, you can index all
//! the arrays into 1d axis, and then call reduce on axis 0.

#![allow(clippy::needless_range_loop)]

use super::{AllocateZeros, Cpu, ForEachBroadcast1};
use crate::arrays::CountElements;

/// Reduce the `I`th axis of `T`. For example given T of shape (M, N, O),
/// you can reduce:
/// 1. the 0th axis (M) to give a shape of (N, O)
/// 2. the 1st axis (N) to give a shape of (M, O)
/// 3. the 2nd axis (O) to give a shape of (M, N)
pub trait ReduceAxis<T: CountElements, Reduced: CountElements, const I: isize>:
    AllocateZeros + ForEachBroadcast1<Reduced, T, I>
{
    fn reduce_into<F>(inp: &T, out: &mut Reduced, f: F)
    where
        F: Copy + FnMut(T::Dtype, T::Dtype) -> T::Dtype;

    fn reduce<F>(inp: &T, f: F) -> Box<Reduced>
    where
        F: Copy + FnMut(T::Dtype, T::Dtype) -> T::Dtype,
    {
        let mut out: Box<Reduced> = Self::zeros();
        Self::reduce_into(inp, out.as_mut(), f);
        out
    }
}

impl ReduceAxis<f32, f32, 0> for Cpu {
    fn reduce_into<F>(inp: &f32, out: &mut f32, _f: F)
    where
        F: Copy + FnMut(f32, f32) -> f32,
    {
        *out = *inp;
    }
}
impl ReduceAxis<f32, f32, -1> for Cpu {
    fn reduce_into<F>(inp: &f32, out: &mut f32, _f: F)
    where
        F: Copy + FnMut(f32, f32) -> f32,
    {
        *out = *inp;
    }
}

impl<const M: usize> ReduceAxis<[f32; M], f32, 0> for Cpu {
    fn reduce_into<F: FnMut(f32, f32) -> f32>(inp: &[f32; M], out: &mut f32, f: F) {
        *out = inp.iter().cloned().reduce(f).unwrap();
    }
}

impl<const M: usize> ReduceAxis<[f32; M], f32, -1> for Cpu {
    fn reduce_into<F: FnMut(f32, f32) -> f32>(inp: &[f32; M], out: &mut f32, f: F) {
        *out = inp.iter().cloned().reduce(f).unwrap();
    }
}

impl<const M: usize, const N: usize> ReduceAxis<[[f32; N]; M], [f32; N], 0> for Cpu {
    fn reduce_into<F>(inp: &[[f32; N]; M], out: &mut [f32; N], f: F)
    where
        F: Copy + FnMut(f32, f32) -> f32,
    {
        for n in 0..N {
            out[n] = inp.iter().map(|x| x[n]).reduce(f).unwrap();
        }
    }
}

impl<const M: usize, const N: usize, const O: usize>
    ReduceAxis<[[[f32; O]; N]; M], [[f32; O]; N], 0> for Cpu
{
    fn reduce_into<F>(inp: &[[[f32; O]; N]; M], out: &mut [[f32; O]; N], f: F)
    where
        F: Copy + FnMut(f32, f32) -> f32,
    {
        for n in 0..N {
            for o in 0..O {
                out[n][o] = inp.iter().map(|x| x[n][o]).reduce(f).unwrap();
            }
        }
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize>
    ReduceAxis<[[[[f32; P]; O]; N]; M], [[[f32; P]; O]; N], 0> for Cpu
{
    fn reduce_into<F>(inp: &[[[[f32; P]; O]; N]; M], out: &mut [[[f32; P]; O]; N], f: F)
    where
        F: Copy + FnMut(f32, f32) -> f32,
    {
        for n in 0..N {
            for o in 0..O {
                for p in 0..P {
                    out[n][o][p] = inp.iter().map(|x| x[n][o][p]).reduce(f).unwrap();
                }
            }
        }
    }
}

macro_rules! reduce_nonzero_axis {
    ($Idx:expr, $SubIdx:expr, $Reduced:ty, $SubArrTy:ty, {$($Vs:tt),*}) => {
impl<$(const $Vs: usize, )*> ReduceAxis<[$SubArrTy; M], $Reduced, $Idx> for Cpu {
    fn reduce_into<F>(inp: &[$SubArrTy; M], out: &mut $Reduced, f: F)
    where
        F: Copy + FnMut(f32, f32) -> f32,
    {
        for m in 0..M {
            <Self as ReduceAxis<_, _, $SubIdx>>::reduce_into(&inp[m], &mut out[m], f);
        }
    }
}
    };
}

reduce_nonzero_axis!(1, 0, [f32; M], [f32; N], {M, N});
reduce_nonzero_axis!(-1, 0, [f32; M], [f32; N], {M, N});

reduce_nonzero_axis!(1, 0, [[f32; O]; M], [[f32; O]; N], {M, N, O});
reduce_nonzero_axis!(2, 1, [[f32; N]; M], [[f32; O]; N], {M, N, O});
reduce_nonzero_axis!(-1, 1, [[f32; N]; M], [[f32; O]; N], {M, N, O});

reduce_nonzero_axis!(1, 0, [[[f32; P]; O]; M], [[[f32; P]; O]; N], {M, N, O, P});
reduce_nonzero_axis!(2, 1, [[[f32; P]; N]; M], [[[f32; P]; O]; N], {M, N, O, P});
reduce_nonzero_axis!(3, 2, [[[f32; O]; N]; M], [[[f32; P]; O]; N], {M, N, O, P});
reduce_nonzero_axis!(-1, 2, [[[f32; O]; N]; M], [[[f32; P]; O]; N], {M, N, O, P});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::ZeroElements;

    #[test]
    fn test_1d_reductions() {
        let inp = [2., -1., 0., 1., -2.];
        let mut out = ZeroElements::ZEROS;
        <Cpu as ReduceAxis<_, _, 0>>::reduce_into(&inp, &mut out, f32::max);
        assert_eq!(out, 2.);
    }

    #[test]
    fn test_2d_reductions() {
        type T = [[f32; 3]; 2];
        let inp: T = [[-1., 2., -3.], [1., -2., 3.]];

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as ReduceAxis<_, _, 0>>::reduce_into(&inp, &mut out0, f32::max);
        assert_eq!(out0, [1., 2., 3.]);

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as ReduceAxis<_, _, 1>>::reduce_into(&inp, &mut out0, f32::max);
        assert_eq!(out0, [2., 3.]);
    }

    #[test]
    fn test_3d_reductions() {
        type T = [[[f32; 3]; 2]; 2];
        let inp: T = [
            [[-1., 2., -3.], [1., -2., 3.]],
            [[4., -5., -3.], [-1., 6., -3.]],
        ];

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as ReduceAxis<_, _, 0>>::reduce_into(&inp, &mut out0, f32::max);
        assert_eq!(out0, [[4., 2., -3.], [1., 6., 3.]]);

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as ReduceAxis<_, _, 1>>::reduce_into(&inp, &mut out0, f32::max);
        assert_eq!(out0, [[1., 2., 3.], [4., 6., -3.]]);

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as ReduceAxis<_, _, 2>>::reduce_into(&inp, &mut out0, f32::max);
        assert_eq!(out0, [[2., 3.], [4., 6.]]);
    }

    #[test]
    fn test_4d_reductions() {
        type T = [[[[f32; 3]; 2]; 3]; 2];
        let inp: T = [
            [
                [[-1., 2., -3.], [1., -2., 3.]],
                [[4., -5., -3.], [-1., 6., -3.]],
                [[-2., 3., 4.], [-6., -7., 3.]],
            ],
            [
                [[1., -2., 3.], [-1., -2., -3.]],
                [[-4., 5., 3.], [-1., -6., -3.]],
                [[2., -3., -4.], [-6., 7., -3.]],
            ],
        ];

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as ReduceAxis<_, _, 0>>::reduce_into(&inp, &mut out0, f32::max);
        assert_eq!(
            out0,
            [
                [[1., 2., 3.], [1., -2., 3.]],
                [[4., 5., 3.], [-1., 6., -3.]],
                [[2., 3., 4.], [-6., 7., 3.]]
            ]
        );

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as ReduceAxis<_, _, 1>>::reduce_into(&inp, &mut out0, f32::max);
        assert_eq!(
            out0,
            [[[4., 3., 4.], [1., 6., 3.]], [[2., 5., 3.], [-1., 7., -3.]]]
        );

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as ReduceAxis<_, _, 2>>::reduce_into(&inp, &mut out0, f32::max);
        assert_eq!(
            out0,
            [
                [[1., 2., 3.], [4., 6., -3.], [-2., 3., 4.]],
                [[1., -2., 3.], [-1., 5., 3.], [2., 7., -3.]]
            ]
        );

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as ReduceAxis<_, _, 3>>::reduce_into(&inp, &mut out0, f32::max);
        assert_eq!(
            out0,
            [
                [[2., 3.], [4., 6.], [4., 3.]],
                [[3., -1.], [5., -1.], [2., 7.]]
            ]
        );
    }
}
