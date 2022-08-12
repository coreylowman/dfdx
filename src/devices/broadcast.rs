#![allow(clippy::needless_range_loop)]

use super::{Cpu, ForEachElement};

/// Foreach methods with 1 axis broadcasted. `Sm` is `Lg` with the `I`th axis reduced.
///
/// E.g. `Sm = [f32; M], Lg = [[f32; N]; M], I = 1`
pub trait ForEachBroadcast1<Sm, Lg, const I: isize> {
    fn foreach_mb<F>(a: &mut Sm, b: &Lg, f: &mut F)
    where
        F: FnMut(&mut f32, &f32);
    fn foreach_br<F>(a: &mut Lg, b: &Sm, f: &mut F)
    where
        F: FnMut(&mut f32, &f32);

    fn broadcast_copy(a: &mut Lg, b: &Sm) {
        Self::foreach_br(a, b, &mut |l, s| *l = *s);
    }
    fn broadcast_add(a: &mut Sm, b: &Lg) {
        Self::foreach_mb(a, b, &mut |s, l| *s += *l);
    }
}

/// Foreach methods with 2 axes broadcasted. `Sm` is `Lg` with the `I` and `J` axes reduced.
///
/// E.g. `Sm = [f32; M], Lg = [[[f32; O]; N]; M], I = 1, J = 2`
pub trait ForEachBroadcast2<Sm, Lg, const I: isize, const J: isize> {
    fn foreach_mb<F>(a: &mut Sm, b: &Lg, f: &mut F)
    where
        F: FnMut(&mut f32, &f32);
    fn foreach_br<F>(a: &mut Lg, b: &Sm, f: &mut F)
    where
        F: FnMut(&mut f32, &f32);

    fn broadcast_copy(a: &mut Lg, b: &Sm) {
        Self::foreach_br(a, b, &mut |l, s| *l = *s);
    }
    fn broadcast_add(a: &mut Sm, b: &Lg) {
        Self::foreach_mb(a, b, &mut |s, l| *s += *l);
    }
}

pub trait ForEachBroadcast3<Sm, Lg, const I: isize, const J: isize, const K: isize> {
    fn foreach_mb<F>(a: &mut Sm, b: &Lg, f: &mut F)
    where
        F: FnMut(&mut f32, &f32);
    fn foreach_br<F>(a: &mut Lg, b: &Sm, f: &mut F)
    where
        F: FnMut(&mut f32, &f32);

    fn broadcast_copy(a: &mut Lg, b: &Sm) {
        Self::foreach_br(a, b, &mut |l, s| *l = *s);
    }
    fn broadcast_add(a: &mut Sm, b: &Lg) {
        Self::foreach_mb(a, b, &mut |s, l| *s += *l);
    }
}

pub trait ForEachBroadcast4<Sm, Lg, const I: isize, const J: isize, const K: isize, const L: isize>
{
    fn foreach_mb<F>(a: &mut Sm, b: &Lg, f: &mut F)
    where
        F: FnMut(&mut f32, &f32);
    fn foreach_br<F>(a: &mut Lg, b: &Sm, f: &mut F)
    where
        F: FnMut(&mut f32, &f32);

    fn broadcast_copy(a: &mut Lg, b: &Sm) {
        Self::foreach_br(a, b, &mut |l, s| *l = *s);
    }
    fn broadcast_add(a: &mut Sm, b: &Lg) {
        Self::foreach_mb(a, b, &mut |s, l| *s += *l);
    }
}

macro_rules! broadcast_impl_reduce_lrg {
    ($Sm:ty, $Lg:ty, $Trait:tt, [$($Axes:expr),*], ForEachElement, [$($SubAxes:expr),*], {$($Dims:tt),*}) => {
impl<$(const $Dims: usize, )*> $Trait<$Sm, $Lg, $($Axes),*> for Cpu {
    fn foreach_mb<F: FnMut(&mut f32, &f32)>(a: &mut $Sm, b: &$Lg, f: &mut F) {
        for m in 0..M { <Self as ForEachElement<_>>::foreach_mr(a, &b[m], f); }
    }

    fn foreach_br<F: FnMut(&mut f32, &f32)>(a: &mut $Lg, b: &$Sm, f: &mut F) {
        for m in 0..M { <Self as ForEachElement<_>>::foreach_mr(&mut a[m], b, f); }
    }
}
    };
    ($Sm:ty, $Lg:ty, $Trait:tt, [$($Axes:expr),*], $SubTrait:tt, [$($SubAxes:expr),*], {$($Dims:tt),*}) => {
impl<$(const $Dims: usize, )*> $Trait<$Sm, $Lg, $($Axes),*> for Cpu {
    fn foreach_mb<F: FnMut(&mut f32, &f32)>(a: &mut $Sm, b: &$Lg, f: &mut F) {
        for m in 0..M { <Self as $SubTrait<_, _, $($SubAxes),*>>::foreach_mb(a, &b[m], f); }
    }

    fn foreach_br<F: FnMut(&mut f32, &f32)>(a: &mut $Lg, b: &$Sm, f: &mut F) {
        for m in 0..M { <Self as $SubTrait<_, _, $($SubAxes),*>>::foreach_br(&mut a[m], b, f); }
    }
}
    };

}

macro_rules! broadcast_impl_reduce_bth {
    ($Sm:ty, $Lg:ty, $Trait:tt, [$($Axes:expr),*], $SubTrait:tt, [$($SubAxes:expr),*], {$($Dims:tt),*}) => {
impl<$(const $Dims: usize, )*> $Trait<$Sm, $Lg, $($Axes),*> for Cpu {
    fn foreach_mb<F: FnMut(&mut f32, &f32)>(a: &mut $Sm, b: &$Lg, f: &mut F) {
        for m in 0..M { <Self as $SubTrait<_, _, $($SubAxes),*>>::foreach_mb(&mut a[m], &b[m], f); }
    }

    fn foreach_br<F: FnMut(&mut f32, &f32)>(a: &mut $Lg, b: &$Sm, f: &mut F) {
        for m in 0..M { <Self as $SubTrait<_, _, $($SubAxes),*>>::foreach_br(&mut a[m], &b[m], f); }
    }
}
    };
}

impl ForEachBroadcast1<f32, f32, 0> for Cpu {
    fn foreach_br<F>(a: &mut f32, b: &f32, f: &mut F)
    where
        F: FnMut(&mut f32, &f32),
    {
        f(a, b);
    }
    fn foreach_mb<F>(a: &mut f32, b: &f32, f: &mut F)
    where
        F: FnMut(&mut f32, &f32),
    {
        f(a, b);
    }
}
impl ForEachBroadcast1<f32, f32, -1> for Cpu {
    fn foreach_br<F>(a: &mut f32, b: &f32, f: &mut F)
    where
        F: FnMut(&mut f32, &f32),
    {
        f(a, b);
    }
    fn foreach_mb<F>(a: &mut f32, b: &f32, f: &mut F)
    where
        F: FnMut(&mut f32, &f32),
    {
        f(a, b);
    }
}

// 0d
#[rustfmt::skip]
broadcast_impl_reduce_lrg!(f32, [f32; M], ForEachBroadcast1, [0], ForEachElement, [], {M});
#[rustfmt::skip]
broadcast_impl_reduce_lrg!(f32, [f32; M], ForEachBroadcast1, [-1], ForEachElement, [], {M});
broadcast_impl_reduce_lrg!(f32, [[f32; N]; M], ForEachBroadcast2, [0, 1], ForEachBroadcast1, [0], {M, N});
broadcast_impl_reduce_lrg!(f32, [[[f32; O]; N]; M], ForEachBroadcast3, [0, 1, 2], ForEachBroadcast2, [0, 1], {M, N, O});
broadcast_impl_reduce_lrg!(f32, [[[[f32; P]; O]; N]; M], ForEachBroadcast4, [0, 1, 2, 3], ForEachBroadcast3, [0, 1, 2], {M, N, O, P});

// 1d -> 2d
broadcast_impl_reduce_lrg!([f32; N], [[f32; N]; M], ForEachBroadcast1, [0], ForEachElement, [], {M, N});
broadcast_impl_reduce_bth!([f32; M], [[f32; N]; M], ForEachBroadcast1, [1], ForEachBroadcast1, [0], {M, N});
broadcast_impl_reduce_bth!([f32; M], [[f32; N]; M], ForEachBroadcast1, [-1], ForEachBroadcast1, [0], {M, N});

// 1d -> 3d
broadcast_impl_reduce_lrg!([f32; O], [[[f32; O]; N]; M], ForEachBroadcast2, [0, 1], ForEachBroadcast1, [0], {M, N, O});
broadcast_impl_reduce_lrg!([f32; N], [[[f32; O]; N]; M], ForEachBroadcast2, [0, 2], ForEachBroadcast1, [1], {M, N, O});
broadcast_impl_reduce_bth!([f32; M], [[[f32; O]; N]; M], ForEachBroadcast2, [1, 2], ForEachBroadcast2, [0, 1], {M, N, O});

// 1d -> 4d
broadcast_impl_reduce_lrg!([f32; P], [[[[f32; P]; O]; N]; M], ForEachBroadcast3, [0, 1, 2], ForEachBroadcast2, [0, 1], {M, N, O, P});
broadcast_impl_reduce_lrg!([f32; O], [[[[f32; P]; O]; N]; M], ForEachBroadcast3, [0, 1, 3], ForEachBroadcast2, [0, 2], {M, N, O, P});
broadcast_impl_reduce_lrg!([f32; N], [[[[f32; P]; O]; N]; M], ForEachBroadcast3, [0, 2, 3], ForEachBroadcast2, [1, 2], {M, N, O, P});
broadcast_impl_reduce_bth!([f32; M], [[[[f32; P]; O]; N]; M], ForEachBroadcast3, [1, 2, 3], ForEachBroadcast3, [0, 1, 2], {M, N, O, P});

// 2d -> 3d
broadcast_impl_reduce_lrg!([[f32; O]; N], [[[f32; O]; N]; M], ForEachBroadcast1, [0], ForEachElement, [], {M, N, O});
broadcast_impl_reduce_bth!([[f32; O]; M], [[[f32; O]; N]; M], ForEachBroadcast1, [1], ForEachBroadcast1, [0], {M, N, O});
broadcast_impl_reduce_bth!([[f32; N]; M], [[[f32; O]; N]; M], ForEachBroadcast1, [2], ForEachBroadcast1, [1], {M, N, O});
broadcast_impl_reduce_bth!([[f32; N]; M], [[[f32; O]; N]; M], ForEachBroadcast1, [-1], ForEachBroadcast1, [1], {M, N, O});

// 2d -> 4d
broadcast_impl_reduce_bth!([[f32; N]; M], [[[[f32; P]; O]; N]; M], ForEachBroadcast2, [2, 3], ForEachBroadcast2, [1, 2], {M, N, O, P});
broadcast_impl_reduce_bth!([[f32; O]; M], [[[[f32; P]; O]; N]; M], ForEachBroadcast2, [1, 3], ForEachBroadcast2, [0, 2], {M, N, O, P});
broadcast_impl_reduce_bth!([[f32; P]; M], [[[[f32; P]; O]; N]; M], ForEachBroadcast2, [1, 2], ForEachBroadcast2, [0, 1], {M, N, O, P});
broadcast_impl_reduce_lrg!([[f32; O]; N], [[[[f32; P]; O]; N]; M], ForEachBroadcast2, [0, 3], ForEachBroadcast1, [2], {M, N, O, P});
broadcast_impl_reduce_lrg!([[f32; P]; N], [[[[f32; P]; O]; N]; M], ForEachBroadcast2, [0, 2], ForEachBroadcast1, [1], {M, N, O, P});
broadcast_impl_reduce_lrg!([[f32; P]; O], [[[[f32; P]; O]; N]; M], ForEachBroadcast2, [0, 1], ForEachBroadcast1, [0], {M, N, O, P});

// 3d -> 4d
broadcast_impl_reduce_lrg!([[[f32; P]; O]; N], [[[[f32; P]; O]; N]; M], ForEachBroadcast1, [0], ForEachElement, [], {M, N, O, P});
broadcast_impl_reduce_bth!([[[f32; P]; O]; M], [[[[f32; P]; O]; N]; M], ForEachBroadcast1, [1], ForEachBroadcast1, [0], {M, N, O, P});
broadcast_impl_reduce_bth!([[[f32; P]; N]; M], [[[[f32; P]; O]; N]; M], ForEachBroadcast1, [2], ForEachBroadcast1, [1], {M, N, O, P});
broadcast_impl_reduce_bth!([[[f32; O]; N]; M], [[[[f32; P]; O]; N]; M], ForEachBroadcast1, [3], ForEachBroadcast1, [2], {M, N, O, P});
broadcast_impl_reduce_bth!([[[f32; O]; N]; M], [[[[f32; P]; O]; N]; M], ForEachBroadcast1, [-1], ForEachBroadcast1, [2], {M, N, O, P});
