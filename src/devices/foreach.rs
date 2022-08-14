use super::{AllocateZeros, Cpu};
use crate::arrays::{CountElements, MultiDimensional};

/// Apply generic function to various forms/numbers of ndarrays.
///
/// The various versions that exist are:
/// - [ForEachElement::foreach_m()], which takes 1 mut array.
/// - [ForEachElement::foreach_mr()], which takes 1 mut array and 1 ref array
/// - [ForEachElement::foreach_mm()], which takes 2 mut arrays
/// - [ForEachElement::foreach_mrr()], which takes 1 mut array and 2 ref arrays
/// - [ForEachElement::foreach_mmm()], which takes 3 mut arrays
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
        for (a_i, b_i) in a.iter_mut().zip(b.iter_mut()) {
            Self::foreach_mm(a_i, b_i, f);
        }
    }

    fn foreach_mr<F>(a: &mut [T; M], b: &[T; M], f: &mut F)
    where
        F: FnMut(&mut T::Dtype, &T::Dtype),
    {
        for (a_i, b_i) in a.iter_mut().zip(b.iter()) {
            Self::foreach_mr(a_i, b_i, f);
        }
    }

    fn foreach_mmm<F>(a: &mut [T; M], b: &mut [T; M], c: &mut [T; M], f: &mut F)
    where
        F: FnMut(&mut T::Dtype, &mut T::Dtype, &mut T::Dtype),
    {
        for (a_i, (b_i, c_i)) in a.iter_mut().zip(b.iter_mut().zip(c.iter_mut())) {
            Self::foreach_mmm(a_i, b_i, c_i, f);
        }
    }

    fn foreach_mrr<F>(a: &mut [T; M], b: &[T; M], c: &[T; M], f: &mut F)
    where
        F: FnMut(&mut T::Dtype, &T::Dtype, &T::Dtype),
    {
        for (a_i, (b_i, c_i)) in a.iter_mut().zip(b.iter().zip(c.iter())) {
            Self::foreach_mrr(a_i, b_i, c_i, f);
        }
    }
}

/// Singifies that the underlying type should be broadcasted in some capacity.
/// Used with [BroadcastForEach]
pub struct Broadcast<'a, T>(pub &'a T);

/// Signifies that the underyling type should be broadcasted in some capacity.
/// Used with [BroadcastForEach]
pub struct BroadcastMut<'a, T>(pub &'a mut T);

/// Apply generic function to the whole last dimension of ndarrays.
///
/// Versions include:
/// - [ForEachLast::foreachlast_mb], which operates on 1 mut array, and 1 broadcasted ref array.
/// - [ForEachLast::foreachlast_brb], which operates on 1 broadcasted mut array, 1 ref array, and 1 broadcasted ref array.
pub trait ForEachLast<O: CountElements, L: CountElements + MultiDimensional, R: CountElements> {
    /// Applies `f` to the last dimension of `l`, where `r` is broadcasted to be the same size as `l`.
    fn foreachlast_mb<F>(l: &mut L, r: Broadcast<R>, f: &mut F)
    where
        F: FnMut(&mut L::LastDim, &R::Dtype);

    /// Applies `f` to the last dimension of `l`, where `out` and `r` are broadcasted to be the same size as `l`.
    fn foreachlast_brb<F>(out: BroadcastMut<O>, l: &L, r: Broadcast<R>, f: &mut F)
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

    fn foreachlast_brb<F>(o: BroadcastMut<f32>, l: &[f32; M], r: Broadcast<usize>, f: &mut F)
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

    fn foreachlast_brb<F>(o: BroadcastMut<[O; M]>, l: &[L; M], r: Broadcast<[R; M]>, f: &mut F)
    where
        F: FnMut(&mut O::Dtype, &L::LastDim, &R::Dtype),
    {
        for (o_i, (l_i, r_i)) in o.0.iter_mut().zip(l.iter().zip(r.0.iter())) {
            Self::foreachlast_brb(BroadcastMut(o_i), l_i, Broadcast(r_i), f)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foreach_m() {
        let mut a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        Cpu::foreach_m(&mut a, &mut |v| {
            *v += 1.0;
        });
        assert_eq!(a, [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]);
    }

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
    fn test_1d_foreachlast_bm() {
        for i in 0..3 {
            let mut l = [0.0, 0.0, 0.0];
            Cpu::foreachlast_mb(&mut l, Broadcast(&i), &mut |a, b| {
                a[*b] = 1.0;
            });
            assert_eq!(l[i], 1.0);
        }
    }

    #[test]
    fn test_2d_foreachlast_bm() {
        let mut l = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        Cpu::foreachlast_mb(&mut l, Broadcast(&[1, 2]), &mut |a, b| {
            a[*b] = 1.0;
        });
        assert_eq!(l, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn test_1d_foreachlast_brb() {
        let l = [1.0, 2.0, 3.0];
        for i in 0..3 {
            let mut o = 0.0;
            Cpu::foreachlast_brb(BroadcastMut(&mut o), &l, Broadcast(&i), &mut |a, b, c| {
                *a = b[*c];
            });
            assert_eq!(o, l[i]);
        }
    }

    #[test]
    fn test_2d_foreachlast_brb() {
        let l = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let r = [2, 0, 1];
        let mut o = [0.0; 3];
        Cpu::foreachlast_brb(BroadcastMut(&mut o), &l, Broadcast(&r), &mut |a, b, c| {
            *a = b[*c];
        });
        assert_eq!(o, [3.0, 4.0, 8.0]);
    }
}
