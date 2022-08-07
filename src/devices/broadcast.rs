//! Implementation of broadcasting any single axis with foreach.
//!
//! # Implementation details
//!
//! If the 0th axis (commonly `M` items) is broadcasted, then each method does a loop
//! over `M` items, and calls the non-broadcasted version of the functions. For
//! example `foreach_mb(&mut [[f32; 3]; 5], &[f32; 3])` would loop 5 times,
//! indexing `l` each of the 5, passing `r` without indexing,
//! and calling `foreach_mr(&mut [f32; 3], &[f32; 3])`.
//!
//! If the broadcasted axis is not 0, then you need to loop `M` times, indexing
//! everything (because everything has `M` as the first dimension), and then
//! calling the same method but with the broadcasted axis reduced by 1.
//! For example `ForEachAxis<1>::foreach_mb` would calling `ForEachAxis<0>::foreach_mb`
//! inside the loop.

use super::{Cpu, ForEachElement, ReduceAxis};
use crate::arrays::CountElements;

/// Apply a function to arrays where one argument has its `I`th axis reduced.
pub trait ForEachAxis<L: CountElements, const I: isize>: ReduceAxis<L, I> {
    /// Applies `f` to each element of `l` and `r`, where `r` is broadcasted to be the same size as `l`.
    fn foreach_mb<F>(l: &mut L, r: &Self::Reduced, f: &mut F)
    where
        F: FnMut(&mut L::Dtype, &L::Dtype);

    /// Applies `f` to each element of `out`, `l`, and `r`, where `r` is broadcasted to be the same size as `out` and `l`.
    fn foreach_mrb<F>(out: &mut L, l: &L, r: &Self::Reduced, f: &mut F)
    where
        F: FnMut(&mut L::Dtype, &L::Dtype, &L::Dtype);

    /// Applies `f` to each element of `out`, `l`, and `r`, where `out` is broadcasted to be the same size as `l` and `r`.
    fn foreach_brr<F>(out: &mut Self::Reduced, l: &L, r: &L, f: &mut F)
    where
        F: FnMut(&mut L::Dtype, &L::Dtype, &L::Dtype);
}

impl ForEachAxis<f32, 0> for Cpu {
    fn foreach_mb<F: FnMut(&mut f32, &f32)>(l: &mut f32, r: &f32, f: &mut F) {
        f(l, r)
    }
    fn foreach_mrb<F: FnMut(&mut f32, &f32, &f32)>(out: &mut f32, l: &f32, r: &f32, f: &mut F) {
        f(out, l, r)
    }
    fn foreach_brr<F: FnMut(&mut f32, &f32, &f32)>(out: &mut f32, l: &f32, r: &f32, f: &mut F) {
        f(out, l, r)
    }
}

macro_rules! foreach_axis {
    (0, $ArrTy:ty, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> ForEachAxis<$ArrTy, 0> for Cpu {
    fn foreach_mb<F>(l: &mut $ArrTy, r: &Self::Reduced, f: &mut F)
    where
        F: FnMut(&mut f32, &f32),
    {
        for i in 0..M {
            Self::foreach_mr(&mut l[i], r, f);
        }
    }

    fn foreach_mrb<F>(out: &mut $ArrTy, l: &$ArrTy, r: &Self::Reduced, f: &mut F)
    where
        F: FnMut(&mut f32, &f32, &f32),
    {
        for i in 0..M {
            Self::foreach_mrr(&mut out[i], &l[i], r, f);
        }
    }

    fn foreach_brr<F>(out: &mut Self::Reduced, l: &$ArrTy, r: &$ArrTy, f: &mut F)
    where
        F: FnMut(&mut f32, &f32, &f32),
    {
        for i in 0..M {
            Self::foreach_mrr(out, &l[i], &r[i], f);
        }
    }
}
    };

    ($Idx:expr, $SubIdx:expr, $SubArrTy:ty, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> ForEachAxis<[$SubArrTy; M], $Idx> for Cpu {
    fn foreach_mb<F>(l: &mut [$SubArrTy; M], r: &Self::Reduced, f: &mut F)
    where
        F: FnMut(&mut f32, &f32),
    {
        for i in 0..M {
            <Self as ForEachAxis<$SubArrTy, $SubIdx>>::foreach_mb(&mut l[i], &r[i], f);
        }
    }

    fn foreach_mrb<F>(out: &mut [$SubArrTy; M], l: &[$SubArrTy; M], r: &Self::Reduced, f: &mut F)
    where
        F: FnMut(&mut f32, &f32, &f32),
    {
        for i in 0..M {
            <Self as ForEachAxis<$SubArrTy, $SubIdx>>::foreach_mrb(&mut out[i], &l[i], &r[i], f);
        }
    }

    fn foreach_brr<F>(out: &mut Self::Reduced, l: &[$SubArrTy; M], r: &[$SubArrTy; M], f: &mut F)
    where
        F: FnMut(&mut f32, &f32, &f32),
    {
        for i in 0..M {
            <Self as ForEachAxis<$SubArrTy, $SubIdx>>::foreach_brr(&mut out[i], &l[i], &r[i], f);
        }
    }
}
    };
}

foreach_axis!(0, [f32; M], [M]);

foreach_axis!(0, [[f32; N]; M], [M, N]);
foreach_axis!(1, 0, [f32; N], [M, N]);

foreach_axis!(0, [[[f32; O]; N]; M], [M, N, O]);
foreach_axis!(1, 0, [[f32; O]; N], [M, N, O]);
foreach_axis!(2, 1, [[f32; O]; N], [M, N, O]);

foreach_axis!(0, [[[[f32; P]; O]; N]; M], [M, N, O, P]);
foreach_axis!(1, 0, [[[f32; P]; O]; N], [M, N, O, P]);
foreach_axis!(2, 1, [[[f32; P]; O]; N], [M, N, O, P]);
foreach_axis!(3, 2, [[[f32; P]; O]; N], [M, N, O, P]);

#[cfg(test)]
mod tests {
    use rand::rngs::ThreadRng;
    use rand::{thread_rng, Rng};

    use super::super::FillElements;
    use super::*;
    use crate::arrays::ZeroElements;

    fn gen<T: CountElements<Dtype = f32>>(rng: &mut ThreadRng) -> Box<T>
    where
        Cpu: FillElements<T>,
    {
        Cpu::filled(&mut |v| *v = rng.gen_range(-1.0..1.0))
    }

    #[test]
    fn test_0d_foreachaxis() {
        type T = f32;

        let mut rng = thread_rng();

        let inp: Box<f32> = gen(&mut rng);
        let mut out = ZeroElements::ZEROS;
        <Cpu as ForEachAxis<T, 0>>::foreach_mb(&mut out, &inp, &mut |a, b| *a += b);
        assert_eq!(out, *inp);
    }

    #[test]
    fn test_1d_foreachaxis() {
        type T = [f32; 3];

        let mut rng = thread_rng();

        let inp: Box<f32> = gen(&mut rng);
        let mut out: T = ZeroElements::ZEROS;
        <Cpu as ForEachAxis<T, 0>>::foreach_mb(&mut out, inp.as_ref(), &mut |a, b| *a += b);
        assert_eq!(out, [*inp; 3]);
    }

    #[test]
    fn test_2d_foreachaxis() {
        type T = [[f32; 3]; 2];

        let mut rng = thread_rng();

        let inp: Box<[f32; 3]> = gen(&mut rng);
        let mut out: T = ZeroElements::ZEROS;
        <Cpu as ForEachAxis<T, 0>>::foreach_mb(&mut out, &inp, &mut |a, b| *a += b);
        assert_eq!(out, [*inp; 2]);

        let inp: Box<[f32; 2]> = gen(&mut rng);
        let mut out: T = ZeroElements::ZEROS;
        <Cpu as ForEachAxis<T, 1>>::foreach_mb(&mut out, &inp, &mut |a, b| *a += b);
        assert_eq!(out, [[inp[0]; 3], [inp[1]; 3]]);
    }

    #[test]
    fn test_3d_foreachaxis() {
        type T = [[[f32; 3]; 2]; 3];

        let mut rng = thread_rng();

        let inp: Box<[[f32; 3]; 2]> = gen(&mut rng);
        let mut out: T = ZeroElements::ZEROS;
        <Cpu as ForEachAxis<T, 0>>::foreach_mb(&mut out, &inp, &mut |a, b| *a += b);
        assert_eq!(out, [*inp; 3]);

        let inp: Box<[[f32; 3]; 3]> = gen(&mut rng);
        let mut out: T = ZeroElements::ZEROS;
        <Cpu as ForEachAxis<T, 1>>::foreach_mb(&mut out, &inp, &mut |a, b| *a += b);
        assert_eq!(out, [[inp[0]; 2], [inp[1]; 2], [inp[2]; 2]]);

        let inp: Box<[[f32; 2]; 3]> = gen(&mut rng);
        let mut out: T = ZeroElements::ZEROS;
        <Cpu as ForEachAxis<T, 2>>::foreach_mb(&mut out, &inp, &mut |a, b| *a += b);
        assert_eq!(
            out,
            [
                [[inp[0][0]; 3], [inp[0][1]; 3]],
                [[inp[1][0]; 3], [inp[1][1]; 3]],
                [[inp[2][0]; 3], [inp[2][1]; 3]]
            ]
        );
    }

    #[test]
    fn test_4d_foreachaxis() {
        type T = [[[[f32; 1]; 2]; 1]; 2];

        let mut rng = thread_rng();

        let inp: Box<[[[f32; 1]; 2]; 1]> = gen(&mut rng);
        let mut out: T = ZeroElements::ZEROS;
        <Cpu as ForEachAxis<T, 0>>::foreach_mb(&mut out, &inp, &mut |a, b| *a += b);
        assert_eq!(out, [*inp; 2]);

        let inp: Box<[[[f32; 1]; 2]; 2]> = gen(&mut rng);
        let mut out: T = ZeroElements::ZEROS;
        <Cpu as ForEachAxis<T, 1>>::foreach_mb(&mut out, &inp, &mut |a, b| *a += b);
        assert_eq!(out, [[inp[0]], [inp[1]]]);

        let inp: Box<[[[f32; 1]; 1]; 2]> = gen(&mut rng);
        let mut out: T = ZeroElements::ZEROS;
        <Cpu as ForEachAxis<T, 2>>::foreach_mb(&mut out, &inp, &mut |a, b| *a += b);
        assert_eq!(out, [[[inp[0][0]; 2]], [[inp[1][0]; 2]]]);

        let inp: Box<[[[f32; 2]; 1]; 2]> = gen(&mut rng);
        let mut out: T = ZeroElements::ZEROS;
        <Cpu as ForEachAxis<T, 3>>::foreach_mb(&mut out, &inp, &mut |a, b| *a += b);
        assert_eq!(
            out,
            [
                [[[inp[0][0][0]], [inp[0][0][1]]]],
                [[[inp[1][0][0]], [inp[1][0][1]]]],
            ]
        );
    }
}
