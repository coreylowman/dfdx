use super::{AllocateZeros, Cpu};
use crate::arrays::{CountElements, MultiDimensional};

/// Iterate over various versions of two or three Nd arrays at the same time, and apply a generic function to them.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let mut a = [[0.0; 3]; 2];
/// let b = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// Cpu::foreach_mr(&mut a, &b, &mut |x, y| {
///     *x = 2.0 * y;
/// });
/// assert_eq!(a, [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]);
/// ```
///
/// The various versions that exist are:
/// - `foreach_mm`, which takes 2 mutable arrays
/// - `foreach_mr`, which takes 1 mutable and 1 non-mutable array
/// - `foreach_mmm`, which takes 3 mutable arrays
/// - `foreach_mrr`, which takes 1 mutable and 2 non-mutable arrays
pub trait ForEachElement<T: CountElements>: AllocateZeros {
    /// Mutate elements of `a` by applying `f` to all elements of a.
    fn foreach_m<F: FnMut(&mut T::Dtype)>(a: &mut T, f: &mut F);

    /// Mutate elements of `a` by applying `f` to all elements of (a, b).
    /// `mr` stands for mut ref
    fn foreach_mr<F>(a: &mut T, b: &T, f: &mut F)
    where
        F: FnMut(&mut T::Dtype, &T::Dtype);

    /// Mutate elements of `a` and `b` by applying `f` to all elements of (a, b).
    /// `mm` stands for mut mut
    fn foreach_mm<F>(a: &mut T, b: &mut T, f: &mut F)
    where
        F: FnMut(&mut T::Dtype, &mut T::Dtype);

    /// Mutate elements of `a`, `b`, and `c` by applying `f` to all elements of (a, b, c).
    /// `mmm` stands for mut mut mut
    fn foreach_mmm<F>(a: &mut T, b: &mut T, c: &mut T, f: &mut F)
    where
        F: FnMut(&mut T::Dtype, &mut T::Dtype, &mut T::Dtype);

    /// Mutate elements of `a` by applying `f` to all elements of (a, b, c).
    /// `mrr` stands for mut ref ref
    fn foreach_mrr<F>(a: &mut T, b: &T, c: &T, f: &mut F)
    where
        F: FnMut(&mut T::Dtype, &T::Dtype, &T::Dtype);
}

impl ForEachElement<f32> for Cpu {
    fn foreach_m<F: FnMut(&mut <f32 as CountElements>::Dtype)>(a: &mut f32, f: &mut F) {
        f(a)
    }

    fn foreach_mm<F: FnMut(&mut f32, &mut f32)>(a: &mut f32, b: &mut f32, f: &mut F) {
        f(a, b)
    }

    fn foreach_mr<F: FnMut(&mut f32, &f32)>(a: &mut f32, b: &f32, f: &mut F) {
        f(a, b)
    }

    fn foreach_mmm<F>(a: &mut f32, b: &mut f32, c: &mut f32, f: &mut F)
    where
        F: FnMut(&mut f32, &mut f32, &mut f32),
    {
        f(a, b, c)
    }

    fn foreach_mrr<F>(a: &mut f32, b: &f32, c: &f32, f: &mut F)
    where
        F: FnMut(&mut f32, &f32, &f32),
    {
        f(a, b, c)
    }
}

impl<T: CountElements, const M: usize> ForEachElement<[T; M]> for Cpu
where
    Self: ForEachElement<T>,
{
    fn foreach_m<F: FnMut(&mut <[T; M] as CountElements>::Dtype)>(a: &mut [T; M], f: &mut F) {
        for a_i in a.iter_mut() {
            Self::foreach_m(a_i, f);
        }
    }

    fn foreach_mm<F>(a: &mut [T; M], b: &mut [T; M], f: &mut F)
    where
        F: FnMut(&mut T::Dtype, &mut T::Dtype),
    {
        for i in 0..M {
            Self::foreach_mm(&mut a[i], &mut b[i], f);
        }
    }

    fn foreach_mr<F>(a: &mut [T; M], b: &[T; M], f: &mut F)
    where
        F: FnMut(&mut T::Dtype, &T::Dtype),
    {
        for i in 0..M {
            Self::foreach_mr(&mut a[i], &b[i], f);
        }
    }

    fn foreach_mmm<F>(a: &mut [T; M], b: &mut [T; M], c: &mut [T; M], f: &mut F)
    where
        F: FnMut(&mut T::Dtype, &mut T::Dtype, &mut T::Dtype),
    {
        for i in 0..M {
            Self::foreach_mmm(&mut a[i], &mut b[i], &mut c[i], f);
        }
    }

    fn foreach_mrr<F>(a: &mut [T; M], b: &[T; M], c: &[T; M], f: &mut F)
    where
        F: FnMut(&mut T::Dtype, &T::Dtype, &T::Dtype),
    {
        for i in 0..M {
            Self::foreach_mrr(&mut a[i], &b[i], &c[i], f);
        }
    }
}

pub struct Broadcast<'a, T>(pub &'a T);

pub struct BroadcastMut<'a, T>(pub &'a mut T);

pub trait BroadcastForEach<L: CountElements, R: CountElements>: AllocateZeros {
    fn foreach_mb<F>(l: &mut L, r: Broadcast<R>, f: &mut F)
    where
        F: FnMut(&mut L::Dtype, &R::Dtype);

    fn foreach_mrb<F>(out: &mut L, l: &L, r: Broadcast<R>, f: &mut F)
    where
        F: FnMut(&mut L::Dtype, &L::Dtype, &R::Dtype);

    fn foreach_brr<F>(out: BroadcastMut<R>, l1: &L, l2: &L, f: &mut F)
    where
        F: FnMut(&mut R::Dtype, &L::Dtype, &L::Dtype);
}

impl BroadcastForEach<f32, f32> for Cpu {
    fn foreach_mb<F>(l: &mut f32, r: Broadcast<f32>, f: &mut F)
    where
        F: FnMut(&mut f32, &f32),
    {
        f(l, r.0);
    }

    fn foreach_mrb<F>(out: &mut f32, l: &f32, r: Broadcast<f32>, f: &mut F)
    where
        F: FnMut(&mut f32, &f32, &f32),
    {
        f(out, l, r.0);
    }

    fn foreach_brr<F>(out: BroadcastMut<f32>, l1: &f32, l2: &f32, f: &mut F)
    where
        F: FnMut(&mut f32, &f32, &f32),
    {
        f(out.0, l1, l2);
    }
}

impl<const M: usize> BroadcastForEach<[f32; M], f32> for Cpu {
    fn foreach_mb<F>(l: &mut [f32; M], r: Broadcast<f32>, f: &mut F)
    where
        F: FnMut(&mut f32, &f32),
    {
        for l_i in l.iter_mut() {
            f(l_i, r.0);
        }
    }

    fn foreach_mrb<F>(out: &mut [f32; M], l: &[f32; M], r: Broadcast<f32>, f: &mut F)
    where
        F: FnMut(&mut f32, &f32, &f32),
    {
        for (out_i, l_i) in out.iter_mut().zip(l.iter()) {
            f(out_i, l_i, r.0);
        }
    }

    fn foreach_brr<F>(out: BroadcastMut<f32>, l1: &[f32; M], l2: &[f32; M], f: &mut F)
    where
        F: FnMut(&mut f32, &f32, &f32),
    {
        for (l1_i, l2_i) in l1.iter().zip(l2.iter()) {
            f(out.0, l1_i, l2_i);
        }
    }
}

impl<L: CountElements, R: CountElements, const M: usize> BroadcastForEach<[L; M], [R; M]> for Cpu
where
    Self: BroadcastForEach<L, R>,
{
    fn foreach_mb<F>(l: &mut [L; M], r: Broadcast<[R; M]>, f: &mut F)
    where
        F: FnMut(&mut L::Dtype, &R::Dtype),
    {
        for (l_i, r_i) in l.iter_mut().zip(r.0.iter()) {
            Self::foreach_mb(l_i, Broadcast(r_i), f);
        }
    }

    fn foreach_mrb<F>(out: &mut [L; M], l: &[L; M], r: Broadcast<[R; M]>, f: &mut F)
    where
        F: FnMut(&mut L::Dtype, &L::Dtype, &R::Dtype),
    {
        for i in 0..M {
            Self::foreach_mrb(&mut out[i], &l[i], Broadcast(&r.0[i]), f);
        }
    }

    fn foreach_brr<F>(out: BroadcastMut<[R; M]>, l1: &[L; M], l2: &[L; M], f: &mut F)
    where
        F: FnMut(&mut R::Dtype, &L::Dtype, &L::Dtype),
    {
        for i in 0..M {
            Self::foreach_brr(BroadcastMut(&mut out.0[i]), &l1[i], &l2[i], f);
        }
    }
}

pub trait ForEachLast<O: CountElements, L: CountElements + MultiDimensional, R: CountElements> {
    fn foreachlast_mb<F>(l: &mut L, r: Broadcast<R>, f: &mut F)
    where
        F: FnMut(&mut L::LastDim, &R::Dtype);
    fn foreachlast_mrb<F>(o: BroadcastMut<O>, l: &L, r: Broadcast<R>, f: &mut F)
    where
        F: FnMut(&mut O::Dtype, &L::LastDim, &R::Dtype);
}

impl<const M: usize> ForEachLast<f32, [f32; M], usize> for Cpu {
    fn foreachlast_mb<F>(l: &mut [f32; M], r: Broadcast<usize>, f: &mut F)
    where
        F: FnMut(&mut [f32; M], &usize),
    {
        f(l, r.0);
    }

    fn foreachlast_mrb<F>(o: BroadcastMut<f32>, l: &[f32; M], r: Broadcast<usize>, f: &mut F)
    where
        F: FnMut(&mut f32, &[f32; M], &usize),
    {
        f(o.0, l, r.0);
    }
}

impl<O: CountElements, L: CountElements + MultiDimensional, R: CountElements, const M: usize>
    ForEachLast<[O; M], [L; M], [R; M]> for Cpu
where
    Cpu: ForEachLast<O, L, R>,
{
    fn foreachlast_mb<F>(l: &mut [L; M], r: Broadcast<[R; M]>, f: &mut F)
    where
        F: FnMut(&mut L::LastDim, &<[R; M] as CountElements>::Dtype),
    {
        for (l_i, r_i) in l.iter_mut().zip(r.0.iter()) {
            Self::foreachlast_mb(l_i, Broadcast(r_i), f);
        }
    }

    fn foreachlast_mrb<F>(o: BroadcastMut<[O; M]>, l: &[L; M], r: Broadcast<[R; M]>, f: &mut F)
    where
        F: FnMut(&mut O::Dtype, &L::LastDim, &R::Dtype),
    {
        for i in 0..M {
            Self::foreachlast_mrb(BroadcastMut(&mut o.0[i]), &l[i], Broadcast(&r.0[i]), f)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foreach_mr() {
        let mut a = [[0.0; 3]; 2];
        let b = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        Cpu::foreach_mr(&mut a, &b, &mut |x, y| {
            *x = 2.0 * y;
        });
        assert_eq!(a, [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]);
    }

    #[test]
    fn test_foreach_mm() {
        let mut a = [[0.0; 3]; 2];
        let mut b = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        Cpu::foreach_mm(&mut a, &mut b, &mut |x, y| {
            *x = 2.0 * *y;
            *y = 1.0;
        });
        assert_eq!(a, [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]);
        assert_eq!(b, [[1.0; 3]; 2]);
    }

    #[test]
    fn test_foreach_mrr() {
        let mut a = [[0.0; 3]; 2];
        let b = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let c = [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]];
        Cpu::foreach_mrr(&mut a, &b, &c, &mut |x, y, z| {
            *x = *z + *y;
        });
        assert_eq!(a, [[0.0; 3]; 2]);
    }

    #[test]
    fn test_foreach_mmm() {
        let mut a = [[0.0; 3]; 2];
        let mut b = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut c = [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]];
        Cpu::foreach_mmm(&mut a, &mut b, &mut c, &mut |x, y, z| {
            *x = 0.0;
            *y = 1.0;
            *z = 2.0;
        });
        assert_eq!(a, [[0.0; 3]; 2]);
        assert_eq!(b, [[1.0; 3]; 2]);
        assert_eq!(c, [[2.0; 3]; 2]);
    }

    #[test]
    fn test_1d_foreachlast() {
        let l = [1.0, 2.0, 3.0];
        for i in 0..3 {
            let mut o = 0.0;
            Cpu::foreachlast_mrb(BroadcastMut(&mut o), &l, Broadcast(&i), &mut |a, b, c| {
                *a = b[*c];
            });
            assert_eq!(o, l[i]);
        }
    }

    #[test]
    fn test_2d_foreachlast() {
        let l = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let r = [2, 0, 1];
        let mut o = [0.0; 3];
        Cpu::foreachlast_mrb(BroadcastMut(&mut o), &l, Broadcast(&r), &mut |a, b, c| {
            *a = b[*c];
        });
        assert_eq!(o, [3.0, 4.0, 8.0]);
    }
}
