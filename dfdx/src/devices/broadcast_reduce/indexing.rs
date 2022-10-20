use crate::arrays::{Axes2, Axes3, Axes4, Axis};
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
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i]
    }
}

impl<const M: usize> IndexMut for [f32; M] {
    type Index = usize;
    type Element = f32;
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i]
    }
}

impl<const M: usize, const N: usize> IndexRef for [[f32; N]; M] {
    type Index = [usize; 2];
    type Element = f32;
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i[0]][i[1]]
    }
}

impl<const M: usize, const N: usize> IndexMut for [[f32; N]; M] {
    type Index = [usize; 2];
    type Element = f32;
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i[0]][i[1]]
    }
}

impl<const M: usize, const N: usize, const O: usize> IndexRef for [[[f32; O]; N]; M] {
    type Index = [usize; 3];
    type Element = f32;
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i[0]][i[1]][i[2]]
    }
}

impl<const M: usize, const N: usize, const O: usize> IndexMut for [[[f32; O]; N]; M] {
    type Index = [usize; 3];
    type Element = f32;
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i[0]][i[1]][i[2]]
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize> IndexRef
    for [[[[f32; P]; O]; N]; M]
{
    type Index = [usize; 4];
    type Element = f32;
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i[0]][i[1]][i[2]][i[3]]
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize> IndexMut
    for [[[[f32; P]; O]; N]; M]
{
    type Index = [usize; 4];
    type Element = f32;
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i[0]][i[1]][i[2]][i[3]]
    }
}

macro_rules! impl_bcast {
    ($ArrTy:ty, [$($Idx:expr),*], $AxisTy:ty, $IdxTy:ty, {$($CVars:tt),*}) => {
        impl<'a, $(const $CVars: usize, )*> IndexRef for BroadcastRef<'a, $ArrTy, $AxisTy> {
            type Index = $IdxTy;
            type Element = f32;
            #[allow(unused_variables)]
            fn index_ref(&self, i: Self::Index) -> &Self::Element {
                &self.0 $([i[$Idx]])*
            }
        }
        impl<'a, $(const $CVars: usize, )*> IndexMut for BroadcastMut<'a, $ArrTy, $AxisTy> {
            type Index = $IdxTy;
            type Element = f32;
            #[allow(unused_variables)]
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

// 3d -> 4d
impl_bcast!([[[f32; O]; N]; M], [0, 1, 2], Axis<3>, [usize; 4], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [0, 1, 3], Axis<2>, [usize; 4], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [0, 2, 3], Axis<1>, [usize; 4], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [1, 2, 3], Axis<0>, [usize; 4], {M, N, O});
