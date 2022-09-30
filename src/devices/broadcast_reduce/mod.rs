//! Implementations of both braodcast and reduce with a single trait.
//!
//! This is done via three pieces:
//! 1. [Accumulator], which accumulate a sequence of values into a single value.
//! 2. [BroadcastRef] [BroadcastMut], which broadcast a value along specific axes
//! 3. [indexing::IndexRef] and [indexing::IndexMut], which enable indexing into
//!     values
//!
//! How these work together:
//! 1. The indexing traits are implemented for normal arrays, and also [BroadcastRef]
//! and [BroadcastMut]. This means you can broadcast a value and then index it in the
//! same way as a normal array
//! 2. [accum1d], and the 2-4d versions apply an [Accumulator] to two types that impl
//! [indexing::IndexRef] and [indexing::IndexMut]
//! 3. The macros in this file tie the previous two pieces together.

mod accumulator;
mod indexing;

use super::allocate::AllocateZeros;
use super::fill::FillElements;
use super::Cpu;
use crate::arrays::{AllAxes, Axes2, Axes3, Axes4, Axis, CountElements};
pub use accumulator::*;
use indexing::{BroadcastMut, BroadcastRef};

/// Device level broadcasts & reduces of type `T` along axes `Axes`.
pub trait DeviceReduce<T: CountElements, Axes>:
    FillElements<T> + FillElements<Self::Reduced> + AllocateZeros
{
    /// The smaller type.
    type Reduced: CountElements<Dtype = T::Dtype>;

    /// Reduces `T` into `Self::Reduced` with accumulator `A` without resetting the values in `r`.
    fn reduce_into_no_reset<A: Accumulator<T::Dtype>>(r: &mut Self::Reduced, t: &T);

    /// Broadcasts `Self::Reduced` into `T` with accumulator `A` without resetting the values
    /// in `t`.
    fn broadcast_into_no_reset<A: Accumulator<T::Dtype>>(t: &mut T, r: &Self::Reduced);

    /// Fills `r` with [Accumulator::INIT] before reducing.
    fn reduce_into<A: Accumulator<T::Dtype>>(r: &mut Self::Reduced, t: &T) {
        Self::fill(r, &mut |x| *x = A::INIT);
        Self::reduce_into_no_reset::<A>(r, t);
    }

    /// Fills `t` with [Accumulator::INIT] before broadcasting.
    fn broadcast_into<A: Accumulator<T::Dtype>>(t: &mut T, r: &Self::Reduced) {
        Self::fill(t, &mut |x| *x = A::INIT);
        Self::broadcast_into_no_reset::<A>(t, r);
    }

    /// Allocates `Self::Reduced` then calls [DeviceReduce::reduce_into()]
    fn reduce<A: Accumulator<T::Dtype>>(t: &T) -> Box<Self::Reduced> {
        let mut r: Box<Self::Reduced> = Self::zeros();
        Self::reduce_into::<A>(r.as_mut(), t);
        r
    }
}

macro_rules! impl_reduce {
    ($ArrTy:ty, $AxesTy:ty, $RedTy:ty, $Accum:tt, {$($Const:tt),*}) => {
        impl<$(const $Const: usize, )*> DeviceReduce<$ArrTy, $AxesTy> for Cpu {
            type Reduced = $RedTy;
            fn reduce_into_no_reset<A: Accumulator<f32>>(r: &mut Self::Reduced, t: &$ArrTy) {
                let mut b = BroadcastMut::<_, $AxesTy>::new(r);
                $Accum::<A, _, _, $($Const, )*>(&mut b, t);
            }
            fn broadcast_into_no_reset<A: Accumulator<f32>>(t: &mut $ArrTy, r: &Self::Reduced) {
                let b = BroadcastRef::<_, $AxesTy>::new(r);
                $Accum::<A, _, _, $($Const, )*>(t, &b);
            }
        }
    };
}

impl DeviceReduce<f32, Axis<-1>> for Cpu {
    type Reduced = f32;
    fn reduce_into_no_reset<A: Accumulator<f32>>(r: &mut Self::Reduced, t: &f32) {
        A::accum(r, t);
    }
    fn broadcast_into_no_reset<A: Accumulator<f32>>(t: &mut f32, r: &Self::Reduced) {
        A::accum(t, r);
    }
}

// 1d -> 0d
impl_reduce!([f32; M], Axis<0>, f32, accum1d, { M });
impl_reduce!([f32; M], Axis<-1>, f32, accum1d, { M });

// 2d -> 1d
impl_reduce!([[f32; N]; M], Axis<0>, [f32; N], accum2d, {M, N});
impl_reduce!([[f32; N]; M], Axis<1>, [f32; M], accum2d, {M, N});
impl_reduce!([[f32; N]; M], Axis<-1>, [f32; M], accum2d, {M, N});

// 2d -> 0d
impl_reduce!([[f32; N]; M], Axes2<0, 1>, f32, accum2d, {M, N});

// 3d -> 2d
impl_reduce!([[[f32; O]; N]; M], Axis<0>, [[f32; O]; N], accum3d, {M, N, O});
impl_reduce!([[[f32; O]; N]; M], Axis<1>, [[f32; O]; M], accum3d, {M, N, O});
impl_reduce!([[[f32; O]; N]; M], Axis<2>, [[f32; N]; M], accum3d, {M, N, O});
impl_reduce!([[[f32; O]; N]; M], Axis<-1>, [[f32; N]; M], accum3d, {M, N, O});

// 3d -> 1d
impl_reduce!([[[f32; O]; N]; M], Axes2<0, 1>, [f32; O], accum3d, {M, N, O});
impl_reduce!([[[f32; O]; N]; M], Axes2<0, 2>, [f32; N], accum3d, {M, N, O});
impl_reduce!([[[f32; O]; N]; M], Axes2<1, 2>, [f32; M], accum3d, {M, N, O});

// 3d -> 0d
impl_reduce!([[[f32; O]; N]; M], Axes3<0, 1, 2>, f32, accum3d, {M, N, O});

// 4d -> 3d
impl_reduce!([[[[f32; P]; O]; N]; M], Axis<0>, [[[f32; P]; O]; N], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axis<1>, [[[f32; P]; O]; M], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axis<2>, [[[f32; P]; N]; M], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axis<3>, [[[f32; O]; N]; M], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axis<-1>, [[[f32; O]; N]; M], accum4d, {M, N, O, P});

// 4d -> 2d
impl_reduce!([[[[f32; P]; O]; N]; M], Axes2<0, 1>, [[f32; P]; O], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes2<0, 2>, [[f32; P]; N], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes2<0, 3>, [[f32; O]; N], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes2<1, 2>, [[f32; P]; M], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes2<1, 3>, [[f32; O]; M], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes2<2, 3>, [[f32; N]; M], accum4d, {M, N, O, P});

// 4d -> 1d
impl_reduce!([[[[f32; P]; O]; N]; M], Axes3<0, 1, 2>, [f32; P], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes3<0, 1, 3>, [f32; O], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes3<0, 2, 3>, [f32; N], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes3<1, 2, 3>, [f32; M], accum4d, {M, N, O, P});

// 4d -> 0d
impl_reduce!([[[[f32; P]; O]; N]; M], Axes4<0, 1, 2, 3>, f32, accum4d, {M, N, O, P});

impl DeviceReduce<f32, AllAxes> for Cpu {
    type Reduced = f32;
    fn reduce_into_no_reset<A: Accumulator<f32>>(r: &mut Self::Reduced, t: &f32) {
        A::accum(r, t);
    }
    fn broadcast_into_no_reset<A: Accumulator<f32>>(t: &mut f32, r: &Self::Reduced) {
        A::accum(t, r);
    }
}

impl<T: CountElements, const M: usize> DeviceReduce<[T; M], AllAxes> for Cpu
where
    T::Dtype: CountElements<Dtype = T::Dtype>,
    Self: DeviceReduce<T, AllAxes> + FillElements<[T; M]> + FillElements<T::Dtype>,
{
    type Reduced = <Self as DeviceReduce<T, AllAxes>>::Reduced;
    fn reduce_into_no_reset<A: Accumulator<T::Dtype>>(r: &mut Self::Reduced, t: &[T; M]) {
        for t_i in t.iter() {
            Self::reduce_into_no_reset::<A>(r, t_i);
        }
    }
    fn broadcast_into_no_reset<A: Accumulator<T::Dtype>>(t: &mut [T; M], r: &Self::Reduced) {
        for t_i in t.iter_mut() {
            Self::broadcast_into_no_reset::<A>(t_i, r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::ZeroElements;

    #[test]
    fn test_reduce_all_1d() {
        let t = [1.0, 2.0, 3.0, 4.0];
        let mut r = 2.0;
        <Cpu as DeviceReduce<_, AllAxes>>::reduce_into::<SubAccum>(&mut r, &t);
        assert_eq!(r, -10.0);
    }

    #[test]
    fn test_reduce_all_2d() {
        let t = [[1.0, 2.0, 3.0, 4.0], [5.0, -1.0, 2.0, 0.0]];
        let mut r = 0.0;
        <Cpu as DeviceReduce<_, AllAxes>>::reduce_into::<AddAccum>(&mut r, &t);
        assert_eq!(r, 16.0);
    }

    #[test]
    fn test_reduce_all_3d() {
        let t = [[[1.0, 2.0], [2.0, 3.0]], [[1.0, 0.5], [0.5, 1.0 / 3.0]]];
        let mut r = 0.0;
        <Cpu as DeviceReduce<_, AllAxes>>::reduce_into::<MulAccum>(&mut r, &t);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn test_1d_1axis_reductions() {
        let inp = [2., -1., 0., 1., -2.];
        let mut out = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<0>>>::reduce_into::<MaxAccum>(&mut out, &inp);
        assert_eq!(out, 2.);
    }

    #[test]
    fn test_2d_1axis_reductions() {
        type T = [[f32; 3]; 2];
        let inp: T = [[-1., 2., -3.], [1., -2., 3.]];

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<0>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(out0, [1., 2., 3.]);

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<1>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(out0, [2., 3.]);
    }

    #[test]
    fn test_3d_1axis_reductions() {
        type T = [[[f32; 3]; 2]; 2];
        let inp: T = [
            [[-1., 2., -3.], [1., -2., 3.]],
            [[4., -5., -3.], [-1., 6., -3.]],
        ];

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<0>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(out0, [[4., 2., -3.], [1., 6., 3.]]);

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<1>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(out0, [[1., 2., 3.], [4., 6., -3.]]);

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<2>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(out0, [[2., 3.], [4., 6.]]);
    }

    #[test]
    fn test_4d_1axis_reductions() {
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
        <Cpu as DeviceReduce<_, Axis<0>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(
            out0,
            [
                [[1., 2., 3.], [1., -2., 3.]],
                [[4., 5., 3.], [-1., 6., -3.]],
                [[2., 3., 4.], [-6., 7., 3.]]
            ]
        );

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<1>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(
            out0,
            [[[4., 3., 4.], [1., 6., 3.]], [[2., 5., 3.], [-1., 7., -3.]]]
        );

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<2>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(
            out0,
            [
                [[1., 2., 3.], [4., 6., -3.], [-2., 3., 4.]],
                [[1., -2., 3.], [-1., 5., 3.], [2., 7., -3.]]
            ]
        );

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<3>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(
            out0,
            [
                [[2., 3.], [4., 6.], [4., 3.]],
                [[3., -1.], [5., -1.], [2., 7.]]
            ]
        );
    }

    #[test]
    fn test_reduce_3d_to_1d() {
        let inp = [
            [[-1., 2., -3.], [1., -2., 3.]],
            [[4., -5., -3.], [-1., 6., -3.]],
            [[7., -3., -2.], [0., 3., 5.]],
            [[-7., 3., 2.], [5., -3., -2.]],
        ];

        {
            let mut out: [f32; 3] = ZeroElements::ZEROS;
            <Cpu as DeviceReduce<_, Axes2<0, 1>>>::reduce_into::<MaxAccum>(&mut out, &inp);
            assert_eq!(out, [7., 6., 5.]);
        }

        {
            let mut out: [f32; 2] = ZeroElements::ZEROS;
            <Cpu as DeviceReduce<_, Axes2<0, 2>>>::reduce_into::<MinAccum>(&mut out, &inp);
            assert_eq!(out, [-7., -3.]);
        }

        {
            let mut out: [f32; 4] = ZeroElements::ZEROS;
            <Cpu as DeviceReduce<_, Axes2<1, 2>>>::reduce_into::<AddAccum>(&mut out, &inp);
            assert_eq!(out, [0., -2., 10., -2.]);
        }
    }

    #[test]
    fn test_broadcast_0d_to_1d() {
        let mut a = [-1.0; 3];
        <Cpu as DeviceReduce<_, Axis<0>>>::broadcast_into::<CopyAccum>(&mut a, &1.0);
        assert_eq!(a, [1.0; 3]);

        let mut a = -1.0;
        <Cpu as DeviceReduce<_, Axis<0>>>::reduce_into::<AddAccum>(&mut a, &[1.0, -2.0, 3.0]);
        assert_eq!(a, 2.0);
    }

    #[test]
    fn test_broadcast_1d_to_2d() {
        let mut a = [[0.0; 3]; 2];
        <Cpu as DeviceReduce<_, Axis<0>>>::broadcast_into::<CopyAccum>(&mut a, &[1.0, 2.0, 3.0]);
        assert_eq!(a, [[1.0, 2.0, 3.0]; 2]);

        let mut a = [[0.0; 3]; 2];
        <Cpu as DeviceReduce<_, Axis<1>>>::broadcast_into::<CopyAccum>(&mut a, &[1.0, 2.0]);
        assert_eq!(a, [[1.0; 3], [2.0; 3]]);

        let mut b = [-1.0, 2.0];
        <Cpu as DeviceReduce<_, Axis<1>>>::reduce_into::<AddAccum>(&mut b, &a);
        assert_eq!(b, [3.0, 6.0]);
    }

    #[test]
    fn test_broadcast_1d_to_3d() {
        let mut a = [[[0.0; 3]; 2]; 1];
        <Cpu as DeviceReduce<_, Axes2<0, 1>>>::broadcast_into::<CopyAccum>(
            &mut a,
            &[1.0, 2.0, 3.0],
        );
        assert_eq!(a, [[[1.0, 2.0, 3.0]; 2]]);

        let mut a = [[[0.0; 3]; 2]; 1];
        <Cpu as DeviceReduce<_, Axes2<0, 2>>>::broadcast_into::<CopyAccum>(&mut a, &[1.0, 2.0]);
        assert_eq!(a, [[[1.0; 3], [2.0; 3]]]);

        let mut a = [[[0.0; 3]; 2]; 3];
        <Cpu as DeviceReduce<_, Axes2<1, 2>>>::broadcast_into::<CopyAccum>(
            &mut a,
            &[1.0, 2.0, 3.0],
        );
        assert_eq!(a, [[[1.0; 3]; 2], [[2.0; 3]; 2], [[3.0; 3]; 2]]);
    }

    #[test]
    fn test_broadcast_1d_to_4d() {
        let mut a = [[[[0.0; 3]; 2]; 1]; 4];
        <Cpu as DeviceReduce<_, Axes3<0, 1, 2>>>::broadcast_into::<CopyAccum>(
            &mut a,
            &[1.0, 2.0, 3.0],
        );
        assert_eq!(a, [[[[1.0, 2.0, 3.0]; 2]; 1]; 4]);

        let mut a = [[[[0.0; 3]; 2]; 1]; 4];
        <Cpu as DeviceReduce<_, Axes3<0, 1, 3>>>::broadcast_into::<CopyAccum>(&mut a, &[1.0, 2.0]);
        assert_eq!(a, [[[[1.0; 3], [2.0; 3]]; 1]; 4]);

        let mut a = [[[[0.0; 3]; 2]; 1]; 4];
        <Cpu as DeviceReduce<_, Axes3<0, 2, 3>>>::broadcast_into::<CopyAccum>(&mut a, &[1.0]);
        assert_eq!(a, [[[[1.0; 3]; 2]; 1]; 4]);

        let mut a = [[[[0.0; 3]; 2]; 1]; 4];
        <Cpu as DeviceReduce<_, Axes3<1, 2, 3>>>::broadcast_into::<CopyAccum>(
            &mut a,
            &[1.0, 2.0, 3.0, 4.0],
        );
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
        <Cpu as DeviceReduce<_, Axis<0>>>::broadcast_into::<CopyAccum>(
            &mut a,
            &[[1.0, 2.0], [-1.0, -2.0]],
        );
        assert_eq!(a, [[[1.0, 2.0], [-1.0, -2.0]]]);

        let mut a = [[[0.0; 2]; 2]; 1];
        <Cpu as DeviceReduce<_, Axis<1>>>::broadcast_into::<CopyAccum>(&mut a, &[[1.0, 2.0]]);
        assert_eq!(a, [[[1.0, 2.0], [1.0, 2.0]]]);

        let mut a = [[[0.0; 2]; 2]; 1];
        <Cpu as DeviceReduce<_, Axis<2>>>::broadcast_into::<CopyAccum>(&mut a, &[[1.0, 2.0]]);
        assert_eq!(a, [[[1.0, 1.0], [2.0, 2.0]]]);
    }
}
