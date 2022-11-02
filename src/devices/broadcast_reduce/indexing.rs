use crate::arrays::*;
use std::marker::PhantomData;

/// Broadcasts `&'a T` along `Axes` to enable indexing as a higher dimensional array.
pub(super) struct BroadcastRef<'a, T, Axes>(pub &'a T, PhantomData<*const Axes>);

impl<'a, T, Axes> BroadcastRef<'a, T, Axes> {
    pub fn new(t: &'a T) -> Self {
        Self(t, PhantomData)
    }
}

/// Broadcasts `&'a mut T` along `Axes` to enable indexing as a higher dimensional array.
pub(super) struct BroadcastMut<'a, T, Axes>(pub &'a mut T, PhantomData<*const Axes>);

impl<'a, T, Axes> BroadcastMut<'a, T, Axes> {
    pub fn new(t: &'a mut T) -> Self {
        Self(t, PhantomData)
    }
}

/// Index to get a `&Self::Element`.
pub(super) trait IndexRef {
    type Index;
    type Element;
    fn index_ref(&self, i: Self::Index) -> &Self::Element;
}

/// Index to get a `&mut Self::Element`.
pub(super) trait IndexMut {
    type Index;
    type Element;
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element;
}

impl<const M: usize> IndexRef for [f32; M] {
    type Index = usize;
    type Element = f32;
    #[inline(always)]
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i]
    }
}

impl<const M: usize> IndexMut for [f32; M] {
    type Index = usize;
    type Element = f32;
    #[inline(always)]
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i]
    }
}

impl<const M: usize, const N: usize> IndexRef for [[f32; N]; M] {
    type Index = [usize; 2];
    type Element = f32;
    #[inline(always)]
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i[0]][i[1]]
    }
}

impl<const M: usize, const N: usize> IndexMut for [[f32; N]; M] {
    type Index = [usize; 2];
    type Element = f32;
    #[inline(always)]
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i[0]][i[1]]
    }
}

impl<const M: usize, const N: usize, const O: usize> IndexRef for [[[f32; O]; N]; M] {
    type Index = [usize; 3];
    type Element = f32;
    #[inline(always)]
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i[0]][i[1]][i[2]]
    }
}

impl<const M: usize, const N: usize, const O: usize> IndexMut for [[[f32; O]; N]; M] {
    type Index = [usize; 3];
    type Element = f32;
    #[inline(always)]
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i[0]][i[1]][i[2]]
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize> IndexRef
    for [[[[f32; P]; O]; N]; M]
{
    type Index = [usize; 4];
    type Element = f32;
    #[inline(always)]
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i[0]][i[1]][i[2]][i[3]]
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize> IndexMut
    for [[[[f32; P]; O]; N]; M]
{
    type Index = [usize; 4];
    type Element = f32;
    #[inline(always)]
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i[0]][i[1]][i[2]][i[3]]
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize> IndexRef
    for [[[[[f32; Q]; P]; O]; N]; M]
{
    type Index = [usize; 5];
    type Element = f32;
    #[inline(always)]
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i[0]][i[1]][i[2]][i[3]][i[4]]
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize> IndexMut
    for [[[[[f32; Q]; P]; O]; N]; M]
{
    type Index = [usize; 5];
    type Element = f32;
    #[inline(always)]
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i[0]][i[1]][i[2]][i[3]][i[4]]
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize, const R: usize> IndexRef
    for [[[[[[f32; R]; Q]; P]; O]; N]; M]
{
    type Index = [usize; 6];
    type Element = f32;
    #[inline(always)]
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i[0]][i[1]][i[2]][i[3]][i[4]][i[5]]
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize, const R: usize> IndexMut
    for [[[[[[f32; R]; Q]; P]; O]; N]; M]
{
    type Index = [usize; 6];
    type Element = f32;
    #[inline(always)]
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i[0]][i[1]][i[2]][i[3]][i[4]][i[5]]
    }
}

macro_rules! impl_bcast {
    ($ArrTy:ty, [$($Idx:expr),*], $AxisTy:ty, $IdxTy:ty, {$($CVars:tt),*}) => {
        impl<'a, $(const $CVars: usize, )*> IndexRef for BroadcastRef<'a, $ArrTy, $AxisTy> {
            type Index = $IdxTy;
            type Element = f32;
            #[allow(unused_variables)]
            #[inline(always)]
            fn index_ref(&self, i: Self::Index) -> &Self::Element {
                &self.0 $([i[$Idx]])*
            }
        }
        impl<'a, $(const $CVars: usize, )*> IndexMut for BroadcastMut<'a, $ArrTy, $AxisTy> {
            type Index = $IdxTy;
            type Element = f32;
            #[allow(unused_variables)]
            #[inline(always)]
            fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
                &mut self.0 $([i[$Idx]])*
            }
        }
    };
}

// 0d -> nd
impl_bcast!(f32, [], Axis<0>, usize, {});
impl_bcast!(f32, [], Axes2<0, 1>, [usize; 2], {});
impl_bcast!(f32, [], Axes3<0, 1, 2>, [usize; 3], {});
impl_bcast!(f32, [], Axes4<0, 1, 2, 3>, [usize; 4], {});
impl_bcast!(f32, [], Axes5<0, 1, 2, 3, 4>, [usize; 5], {});

// 1d -> 2d
impl_bcast!([f32; M], [0], Axis<1>, [usize; 2], { M });
impl_bcast!([f32; M], [1], Axis<0>, [usize; 2], { M });

// 1d -> 3d
impl_bcast!([f32; M], [2], Axes2<0, 1>, [usize; 3], { M });
impl_bcast!([f32; M], [1], Axes2<0, 2>, [usize; 3], { M });
impl_bcast!([f32; M], [0], Axes2<1, 2>, [usize; 3], { M });

// 1d -> 4d
impl_bcast!([f32; M], [3], Axes3<0, 1, 2>, [usize; 4], { M });
impl_bcast!([f32; M], [2], Axes3<0, 1, 3>, [usize; 4], { M });
impl_bcast!([f32; M], [1], Axes3<0, 2, 3>, [usize; 4], { M });
impl_bcast!([f32; M], [0], Axes3<1, 2, 3>, [usize; 4], { M });

// 1d -> 5d
impl_bcast!([f32; M], [4], Axes4<0, 1, 2, 3>, [usize; 5], { M });
impl_bcast!([f32; M], [3], Axes4<0, 1, 2, 4>, [usize; 5], { M });
impl_bcast!([f32; M], [2], Axes4<0, 1, 3, 4>, [usize; 5], { M });
impl_bcast!([f32; M], [1], Axes4<0, 2, 3, 4>, [usize; 5], { M });
impl_bcast!([f32; M], [0], Axes4<1, 2, 3, 4>, [usize; 5], { M });

// 2d -> 3d
impl_bcast!([[f32; N]; M], [0, 1], Axis<2>, [usize; 3], {M, N});
impl_bcast!([[f32; N]; M], [0, 2], Axis<1>, [usize; 3], {M, N});
impl_bcast!([[f32; N]; M], [1, 2], Axis<0>, [usize; 3], {M, N});

// 2d -> 4d
impl_bcast!([[f32; N]; M], [2, 3], Axes2<0, 1>, [usize; 4], {M, N});
impl_bcast!([[f32; N]; M], [1, 3], Axes2<0, 2>, [usize; 4], {M, N});
impl_bcast!([[f32; N]; M], [1, 2], Axes2<0, 3>, [usize; 4], {M, N});
impl_bcast!([[f32; N]; M], [0, 3], Axes2<1, 2>, [usize; 4], {M, N});
impl_bcast!([[f32; N]; M], [0, 2], Axes2<1, 3>, [usize; 4], {M, N});
impl_bcast!([[f32; N]; M], [0, 1], Axes2<2, 3>, [usize; 4], {M, N});

// 2d -> 5d
impl_bcast!([[f32; N]; M], [3, 4], Axes3<0, 1, 2>, [usize; 5], {M, N});
impl_bcast!([[f32; N]; M], [2, 4], Axes3<0, 1, 3>, [usize; 5], {M, N});
impl_bcast!([[f32; N]; M], [1, 4], Axes3<0, 2, 3>, [usize; 5], {M, N});
impl_bcast!([[f32; N]; M], [0, 4], Axes3<1, 2, 3>, [usize; 5], {M, N});
impl_bcast!([[f32; N]; M], [2, 3], Axes3<0, 1, 4>, [usize; 5], {M, N});
impl_bcast!([[f32; N]; M], [1, 3], Axes3<0, 2, 4>, [usize; 5], {M, N});
impl_bcast!([[f32; N]; M], [0, 3], Axes3<1, 2, 4>, [usize; 5], {M, N});
impl_bcast!([[f32; N]; M], [1, 2], Axes3<0, 3, 4>, [usize; 5], {M, N});
impl_bcast!([[f32; N]; M], [0, 2], Axes3<1, 3, 4>, [usize; 5], {M, N});
impl_bcast!([[f32; N]; M], [0, 1], Axes3<2, 3, 4>, [usize; 5], {M, N});

// 3d -> 4d
impl_bcast!([[[f32; O]; N]; M], [0, 1, 2], Axis<3>, [usize; 4], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [0, 1, 3], Axis<2>, [usize; 4], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [0, 2, 3], Axis<1>, [usize; 4], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [1, 2, 3], Axis<0>, [usize; 4], {M, N, O});

// 3d -> 5d
impl_bcast!([[[f32; O]; N]; M], [0, 1, 2], Axes2<3, 4>, [usize; 5], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [0, 1, 3], Axes2<2, 4>, [usize; 5], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [0, 1, 4], Axes2<2, 3>, [usize; 5], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [0, 2, 3], Axes2<1, 4>, [usize; 5], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [0, 2, 4], Axes2<1, 3>, [usize; 5], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [0, 3, 4], Axes2<1, 2>, [usize; 5], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [1, 2, 3], Axes2<0, 4>, [usize; 5], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [1, 2, 4], Axes2<0, 3>, [usize; 5], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [1, 3, 4], Axes2<0, 2>, [usize; 5], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [2, 3, 4], Axes2<0, 1>, [usize; 5], {M, N, O});

// 4d -> 5d
impl_bcast!([[[[f32; P]; O]; N]; M], [0, 1, 2, 3], Axis<4>, [usize; 5], {M, N, O, P});
impl_bcast!([[[[f32; P]; O]; N]; M], [0, 1, 2, 4], Axis<3>, [usize; 5], {M, N, O, P});
impl_bcast!([[[[f32; P]; O]; N]; M], [0, 1, 3, 4], Axis<2>, [usize; 5], {M, N, O, P});
impl_bcast!([[[[f32; P]; O]; N]; M], [0, 2, 3, 4], Axis<1>, [usize; 5], {M, N, O, P});
impl_bcast!([[[[f32; P]; O]; N]; M], [1, 2, 3, 4], Axis<0>, [usize; 5], {M, N, O, P});

#[cfg(tensor6d)]
pub use tensor6d::*;

#[cfg(tensor6d)]
mod tensor6d {
    use crate::arrays::*;

    // 0d -> 6d
    impl_bcast!(f32, [], Axes6<0, 1, 2, 3, 4, 5>, [usize; 6], {});

    // 1d -> 6d
    impl_bcast!([f32; M], [5], Axes5<0, 1, 2, 3, 4>, [usize; 6], { M });
    impl_bcast!([f32; M], [4], Axes5<0, 1, 2, 3, 5>, [usize; 6], { M });
    impl_bcast!([f32; M], [3], Axes5<0, 1, 2, 4, 5>, [usize; 6], { M });
    impl_bcast!([f32; M], [2], Axes5<0, 1, 3, 4, 5>, [usize; 6], { M });
    impl_bcast!([f32; M], [1], Axes5<0, 2, 3, 4, 5>, [usize; 6], { M });
    impl_bcast!([f32; M], [0], Axes5<1, 2, 3, 4, 5>, [usize; 6], { M });

    // 2d -> 6d
    impl_bcast!([[f32; N]; M], [4, 5], Axes4<0, 1, 2, 3>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [3, 5], Axes4<0, 1, 2, 4>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [2, 5], Axes4<0, 1, 3, 4>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [1, 5], Axes4<0, 2, 3, 4>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [0, 5], Axes4<1, 2, 3, 4>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [3, 4], Axes4<0, 1, 2, 5>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [2, 4], Axes4<0, 1, 3, 5>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [1, 4], Axes4<0, 2, 3, 5>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [0, 4], Axes4<1, 2, 3, 5>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [2, 3], Axes4<0, 1, 4, 5>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [1, 3], Axes4<0, 2, 4, 5>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [0, 3], Axes4<1, 2, 4, 5>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [1, 2], Axes4<0, 3, 4, 5>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [0, 2], Axes4<1, 3, 4, 5>, [usize; 6], {M, N});
    impl_bcast!([[f32; N]; M], [0, 1], Axes4<2, 3, 4, 5>, [usize; 6], {M, N});

    // 3d -> 6d
    impl_bcast!([[[f32; O]; N]; M], [0, 1, 2], Axes3<3, 4, 5>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [0, 1, 3], Axes3<2, 4, 5>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [0, 1, 4], Axes3<2, 3, 5>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [0, 1, 5], Axes3<2, 3, 4>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [0, 2, 3], Axes3<1, 4, 5>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [0, 2, 4], Axes3<1, 3, 5>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [0, 2, 5], Axes3<1, 3, 4>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [0, 3, 4], Axes3<1, 2, 5>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [0, 3, 5], Axes3<1, 2, 4>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [0, 4, 5], Axes3<1, 2, 3>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [1, 2, 3], Axes3<0, 4, 5>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [1, 2, 4], Axes3<0, 3, 5>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [1, 2, 5], Axes3<0, 3, 4>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [1, 3, 4], Axes3<0, 2, 5>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [1, 3, 5], Axes3<0, 2, 4>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [1, 4, 5], Axes3<0, 2, 3>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [2, 3, 4], Axes3<0, 1, 5>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [2, 3, 5], Axes3<0, 1, 4>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [2, 4, 5], Axes3<0, 1, 3>, [usize; 6], {M, N, O});
    impl_bcast!([[[f32; O]; N]; M], [3, 4, 5], Axes3<0, 1, 2>, [usize; 6], {M, N, O});

    // 4d -> 6d
    impl_bcast!([[[[f32; P]; O]; N]; M], [0, 1, 2, 3], Axes2<4, 5>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [0, 1, 2, 4], Axes2<3, 5>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [0, 1, 2, 5], Axes2<3, 4>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [0, 1, 3, 4], Axes2<2, 5>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [0, 1, 3, 5], Axes2<2, 4>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [0, 1, 4, 5], Axes2<2, 3>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [0, 2, 3, 4], Axes2<1, 5>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [0, 2, 3, 5], Axes2<1, 4>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [0, 2, 4, 5], Axes2<1, 3>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [0, 3, 4, 5], Axes2<1, 2>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [1, 2, 3, 4], Axes2<0, 5>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [1, 2, 3, 5], Axes2<0, 4>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [1, 2, 4, 5], Axes2<0, 3>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [1, 3, 4, 5], Axes2<0, 2>, [usize; 6], {M, N, O, P});
    impl_bcast!([[[[f32; P]; O]; N]; M], [2, 3, 4, 5], Axes2<0, 1>, [usize; 6], {M, N, O, P});

    // 5d -> 6d
    impl_bcast!([[[[[f32; Q]; P]; O]; N]; M], [0, 1, 2, 3, 4], Axis<5>, [usize; 6], {M, N, O, P, Q});
    impl_bcast!([[[[[f32; Q]; P]; O]; N]; M], [0, 1, 2, 3, 5], Axis<4>, [usize; 6], {M, N, O, P, Q});
    impl_bcast!([[[[[f32; Q]; P]; O]; N]; M], [0, 1, 2, 4, 5], Axis<3>, [usize; 6], {M, N, O, P, Q});
    impl_bcast!([[[[[f32; Q]; P]; O]; N]; M], [0, 1, 3, 4, 5], Axis<2>, [usize; 6], {M, N, O, P, Q});
    impl_bcast!([[[[[f32; Q]; P]; O]; N]; M], [0, 2, 3, 4, 5], Axis<1>, [usize; 6], {M, N, O, P, Q});
    impl_bcast!([[[[[f32; Q]; P]; O]; N]; M], [1, 2, 3, 4, 5], Axis<0>, [usize; 6], {M, N, O, P, Q});
}