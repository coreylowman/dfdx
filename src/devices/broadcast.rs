//! TODO add details about implementation details for this for future developers

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

/// Foreach methods with 3 axes broadcasted.
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

/// Foreach methods with 4 axes broadcasted.
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
    fn foreach_br<F: FnMut(&mut f32, &f32)>(a: &mut f32, b: &f32, f: &mut F) {
        f(a, b);
    }
    fn foreach_mb<F: FnMut(&mut f32, &f32)>(a: &mut f32, b: &f32, f: &mut F) {
        f(a, b);
    }
}
impl ForEachBroadcast1<f32, f32, -1> for Cpu {
    fn foreach_br<F: FnMut(&mut f32, &f32)>(a: &mut f32, b: &f32, f: &mut F) {
        f(a, b);
    }
    fn foreach_mb<F: FnMut(&mut f32, &f32)>(a: &mut f32, b: &f32, f: &mut F) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_0d_broadcasts() {
        let _ = <Cpu as ForEachBroadcast1<f32, f32, 0>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast1<f32, f32, -1>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast1<f32, [f32; 5], 0>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast1<f32, [f32; 5], -1>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast2<f32, [[f32; 5]; 2], 0, 1>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast3<f32, [[[f32; 5]; 2]; 4], 0, 1, 2>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast4<f32, [[[[f32; 5]; 2]; 4]; 3], 0, 1, 2, 3>>::broadcast_add;
    }

    #[test]
    fn test_valid_1d_broadcasts() {
        let _ = <Cpu as ForEachBroadcast1<[f32; 3], [[f32; 3]; 5], 0>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast1<[f32; 5], [[f32; 3]; 5], 1>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast1<[f32; 5], [[f32; 3]; 5], -1>>::broadcast_add;

        let _ = <Cpu as ForEachBroadcast2<[f32; 3], [[[f32; 3]; 5]; 7], 0, 1>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast2<[f32; 5], [[[f32; 3]; 5]; 7], 0, 2>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast2<[f32; 7], [[[f32; 3]; 5]; 7], 1, 2>>::broadcast_add;

        let _ =
            <Cpu as ForEachBroadcast3<[f32; 1], [[[[f32; 1]; 3]; 5]; 7], 0, 1, 2>>::broadcast_add;
        let _ =
            <Cpu as ForEachBroadcast3<[f32; 3], [[[[f32; 1]; 3]; 5]; 7], 0, 1, 3>>::broadcast_add;
        let _ =
            <Cpu as ForEachBroadcast3<[f32; 5], [[[[f32; 1]; 3]; 5]; 7], 0, 2, 3>>::broadcast_add;
        let _ =
            <Cpu as ForEachBroadcast3<[f32; 7], [[[[f32; 1]; 3]; 5]; 7], 1, 2, 3>>::broadcast_add;
    }

    #[test]
    fn test_valid_2d_broadcasts() {
        let _ = <Cpu as ForEachBroadcast1<[[f32; 3]; 5], [[[f32; 3]; 5]; 7], 0>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast1<[[f32; 3]; 7], [[[f32; 3]; 5]; 7], 1>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast1<[[f32; 5]; 7], [[[f32; 3]; 5]; 7], 2>>::broadcast_add;

        let _ =
            <Cpu as ForEachBroadcast2<[[f32; 5]; 7], [[[[f32; 1]; 3]; 5]; 7], 2, 3>>::broadcast_add;
        let _ =
            <Cpu as ForEachBroadcast2<[[f32; 3]; 7], [[[[f32; 1]; 3]; 5]; 7], 1, 3>>::broadcast_add;
        let _ =
            <Cpu as ForEachBroadcast2<[[f32; 3]; 5], [[[[f32; 1]; 3]; 5]; 7], 0, 3>>::broadcast_add;
        let _ =
            <Cpu as ForEachBroadcast2<[[f32; 1]; 7], [[[[f32; 1]; 3]; 5]; 7], 1, 2>>::broadcast_add;
        let _ =
            <Cpu as ForEachBroadcast2<[[f32; 1]; 5], [[[[f32; 1]; 3]; 5]; 7], 0, 2>>::broadcast_add;
        let _ =
            <Cpu as ForEachBroadcast2<[[f32; 1]; 3], [[[[f32; 1]; 3]; 5]; 7], 0, 1>>::broadcast_add;
    }

    #[test]
    fn test_valid_3d_broadcasts() {
        let _ = <Cpu as ForEachBroadcast1<[[[f32; 1]; 2]; 3], [[[[f32; 1]; 2]; 3]; 4], 0>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast1<[[[f32; 1]; 2]; 4], [[[[f32; 1]; 2]; 3]; 4], 1>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast1<[[[f32; 1]; 3]; 4], [[[[f32; 1]; 2]; 3]; 4], 2>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast1<[[[f32; 2]; 3]; 4], [[[[f32; 1]; 2]; 3]; 4], 3>>::broadcast_add;
        let _ = <Cpu as ForEachBroadcast1<[[[f32; 2]; 3]; 4], [[[[f32; 1]; 2]; 3]; 4], -1>>::broadcast_add;
    }

    #[test]
    fn test_broadcast1_0d_to_0d() {
        let mut a = -1.0;
        <Cpu as ForEachBroadcast1<_, _, 0>>::broadcast_add(&mut a, &1.0);
        assert_eq!(a, 0.0);

        let mut a = -1.0;
        <Cpu as ForEachBroadcast1<_, _, 0>>::broadcast_copy(&mut a, &1.2);
        assert_eq!(a, 1.2);
    }

    #[test]
    fn test_broadcast_0d_to_1d() {
        let mut a = [-1.0; 3];
        <Cpu as ForEachBroadcast1<_, _, 0>>::broadcast_copy(&mut a, &1.0);
        assert_eq!(a, [1.0; 3]);

        let mut a = -1.0;
        <Cpu as ForEachBroadcast1<_, _, 0>>::broadcast_add(&mut a, &[1.0, -2.0, 3.0]);
        assert_eq!(a, 1.0);
    }

    #[test]
    fn test_broadcast_1d_to_2d() {
        let mut a = [[0.0; 3]; 2];
        <Cpu as ForEachBroadcast1<_, _, 0>>::broadcast_copy(&mut a, &[1.0, 2.0, 3.0]);
        assert_eq!(a, [[1.0, 2.0, 3.0]; 2]);

        let mut a = [[0.0; 3]; 2];
        <Cpu as ForEachBroadcast1<_, _, 1>>::broadcast_copy(&mut a, &[1.0, 2.0]);
        assert_eq!(a, [[1.0; 3], [2.0; 3]]);

        let mut b = [-1.0, 2.0];
        <Cpu as ForEachBroadcast1<_, _, 1>>::broadcast_add(&mut b, &a);
        assert_eq!(b, [2.0, 8.0]);
    }

    #[test]
    fn test_broadcast_1d_to_3d() {
        let mut a = [[[0.0; 3]; 2]; 1];
        <Cpu as ForEachBroadcast2<_, _, 0, 1>>::broadcast_copy(&mut a, &[1.0, 2.0, 3.0]);
        assert_eq!(a, [[[1.0, 2.0, 3.0]; 2]]);

        let mut a = [[[0.0; 3]; 2]; 1];
        <Cpu as ForEachBroadcast2<_, _, 0, 2>>::broadcast_copy(&mut a, &[1.0, 2.0]);
        assert_eq!(a, [[[1.0; 3], [2.0; 3]]]);

        let mut a = [[[0.0; 3]; 2]; 3];
        <Cpu as ForEachBroadcast2<_, _, 1, 2>>::broadcast_copy(&mut a, &[1.0, 2.0, 3.0]);
        assert_eq!(a, [[[1.0; 3]; 2], [[2.0; 3]; 2], [[3.0; 3]; 2]]);
    }

    #[test]
    fn test_broadcast_1d_to_4d() {
        let mut a = [[[[0.0; 3]; 2]; 1]; 4];
        <Cpu as ForEachBroadcast3<_, _, 0, 1, 2>>::broadcast_copy(&mut a, &[1.0, 2.0, 3.0]);
        assert_eq!(a, [[[[1.0, 2.0, 3.0]; 2]; 1]; 4]);

        let mut a = [[[[0.0; 3]; 2]; 1]; 4];
        <Cpu as ForEachBroadcast3<_, _, 0, 1, 3>>::broadcast_copy(&mut a, &[1.0, 2.0]);
        assert_eq!(a, [[[[1.0; 3], [2.0; 3]]; 1]; 4]);

        let mut a = [[[[0.0; 3]; 2]; 1]; 4];
        <Cpu as ForEachBroadcast3<_, _, 0, 2, 3>>::broadcast_copy(&mut a, &[1.0]);
        assert_eq!(a, [[[[1.0; 3]; 2]; 1]; 4]);

        let mut a = [[[[0.0; 3]; 2]; 1]; 4];
        <Cpu as ForEachBroadcast3<_, _, 1, 2, 3>>::broadcast_copy(&mut a, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            a,
            [
                [[[1.0; 3]; 2]; 1],
                [[[2.0; 3]; 2]; 1],
                [[[3.0; 3]; 2]; 1],
                [[[4.0; 3]; 2]; 1]
            ]
        );
    }

    #[test]
    fn test_broadcast_2d_to_3d() {
        let mut a = [[[0.0; 2]; 2]; 1];
        <Cpu as ForEachBroadcast1<_, _, 0>>::broadcast_copy(&mut a, &[[1.0, 2.0], [-1.0, -2.0]]);
        assert_eq!(a, [[[1.0, 2.0], [-1.0, -2.0]]]);

        let mut a = [[[0.0; 2]; 2]; 1];
        <Cpu as ForEachBroadcast1<_, _, 1>>::broadcast_copy(&mut a, &[[1.0, 2.0]]);
        assert_eq!(a, [[[1.0, 2.0], [1.0, 2.0]]]);

        let mut a = [[[0.0; 2]; 2]; 1];
        <Cpu as ForEachBroadcast1<_, _, 2>>::broadcast_copy(&mut a, &[[1.0, 2.0]]);
        assert_eq!(a, [[[1.0, 1.0], [2.0, 2.0]]]);
    }
}
