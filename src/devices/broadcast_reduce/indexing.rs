use crate::devices::axes::*;
use std::marker::PhantomData;

pub(super) struct BroadcastRef<'a, T, Axes>(pub &'a T, PhantomData<*const Axes>);

impl<'a, T, Axes> BroadcastRef<'a, T, Axes> {
    pub fn new(t: &'a T) -> Self {
        Self(t, PhantomData)
    }
}

pub(super) struct BroadcastMut<'a, T, Axes>(pub &'a mut T, PhantomData<*const Axes>);

impl<'a, T, Axes> BroadcastMut<'a, T, Axes> {
    pub fn new(t: &'a mut T) -> Self {
        Self(t, PhantomData)
    }
}

pub(super) trait ElementRef {
    type Index;
    type Element;
    fn index_ref(&self, i: Self::Index) -> &Self::Element;
}

pub(super) trait ElementMut {
    type Index;
    type Element;
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element;
}

impl<const M: usize> ElementRef for [f32; M] {
    type Index = usize;
    type Element = f32;
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i]
    }
}

impl<const M: usize> ElementMut for [f32; M] {
    type Index = usize;
    type Element = f32;
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i]
    }
}

impl<const M: usize, const N: usize> ElementRef for [[f32; N]; M] {
    type Index = [usize; 2];
    type Element = f32;
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i[0]][i[1]]
    }
}

impl<const M: usize, const N: usize> ElementMut for [[f32; N]; M] {
    type Index = [usize; 2];
    type Element = f32;
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i[0]][i[1]]
    }
}

impl<const M: usize, const N: usize, const O: usize> ElementRef for [[[f32; O]; N]; M] {
    type Index = [usize; 3];
    type Element = f32;
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i[0]][i[1]][i[2]]
    }
}

impl<const M: usize, const N: usize, const O: usize> ElementMut for [[[f32; O]; N]; M] {
    type Index = [usize; 3];
    type Element = f32;
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i[0]][i[1]][i[2]]
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize> ElementRef
    for [[[[f32; P]; O]; N]; M]
{
    type Index = [usize; 4];
    type Element = f32;
    fn index_ref(&self, i: Self::Index) -> &Self::Element {
        &self[i[0]][i[1]][i[2]][i[3]]
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize> ElementMut
    for [[[[f32; P]; O]; N]; M]
{
    type Index = [usize; 4];
    type Element = f32;
    fn index_mut(&mut self, i: Self::Index) -> &mut Self::Element {
        &mut self[i[0]][i[1]][i[2]][i[3]]
    }
}

macro_rules! impl_bcast {
    ($ArrTy:ty, [], [$I0:expr], {$($CVars:tt),*}) => {
        impl_bcast!($ArrTy, [], Axis<$I0>, usize, {$($CVars),*});
    };
    ($ArrTy:ty, [], [$I0:expr, $I1:expr], {$($CVars:tt),*}) => {
        impl_bcast!($ArrTy, [], Axes2<$I0, $I1>, [usize; 2], {$($CVars),*});
    };
    ($ArrTy:ty, [], [$I0:expr, $I1:expr, $I2:expr], {$($CVars:tt),*}) => {
        impl_bcast!($ArrTy, [], Axes3<$I0, $I1, $I2>, [usize; 3], {$($CVars),*});
    };
    ($ArrTy:ty, [], [$I0:expr, $I1:expr, $I2:expr, $I3:expr], {$($CVars:tt),*}) => {
        impl_bcast!($ArrTy, [], Axes4<$I0, $I1, $I2, $I3>, [usize; 4], {$($CVars),*});
    };
    ($ArrTy:ty, [$I0:expr], [$I1:expr], {$($CVars:tt),*}) => {
        impl_bcast!($ArrTy, [$I0], Axis<$I1>, [usize; 2], {$($CVars),*});
    };
    ($ArrTy:ty, [$I0:expr], [$I1:expr, $I2:expr], {$($CVars:tt),*}) => {
        impl_bcast!($ArrTy, [$I0], Axes2<$I1, $I2>, [usize; 3], {$($CVars),*});
    };
    ($ArrTy:ty, [$I0:expr], [$I1:expr, $I2:expr, $I3:expr], {$($CVars:tt),*}) => {
        impl_bcast!($ArrTy, [$I0], Axes3<$I1, $I2, $I3>, [usize; 4], {$($CVars),*});
    };
    ($ArrTy:ty, [$I0:expr, $I1:expr], [$I2:expr], {$($CVars:tt),*}) => {
        impl_bcast!($ArrTy, [$I0, $I1], Axis<$I2>, [usize; 3], {$($CVars),*});
    };
    ($ArrTy:ty, [$I0:expr, $I1:expr], [$I2:expr, $I3:expr], {$($CVars:tt),*}) => {
        impl_bcast!($ArrTy, [$I0, $I1], Axes2<$I2, $I3>, [usize; 4], {$($CVars),*});
    };
    ($ArrTy:ty, [$I0:expr, $I1:expr, $I2:expr], [$I3:expr], {$($CVars:tt),*}) => {
        impl_bcast!($ArrTy, [$I0, $I1, $I2], Axis<$I3>, [usize; 4], {$($CVars),*});
    };
    ($ArrTy:ty, [$($Idx:expr),*], $AxisTy:ty, $IdxTy:ty, {$($CVars:tt),*}) => {
        impl<'a, $(const $CVars: usize, )*> ElementRef for BroadcastRef<'a, $ArrTy, $AxisTy> {
            type Index = $IdxTy;
            type Element = f32;
            #[allow(unused_variables)]
            fn index_ref(&self, i: Self::Index) -> &Self::Element {
                &self.0 $([i[$Idx]])*
            }
        }
        impl<'a, $(const $CVars: usize, )*> ElementMut for BroadcastMut<'a, $ArrTy, $AxisTy> {
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
impl_bcast!(f32, [], [-1], {});
impl_bcast!(f32, [], [0], {});
impl_bcast!(f32, [], [0, 1], {});
impl_bcast!(f32, [], [0, 1, 2], {});
impl_bcast!(f32, [], [0, 1, 2, 3], {});

// 1d -> 2d
impl_bcast!([f32; M], [0], [-1], { M });
impl_bcast!([f32; M], [0], [1], { M });
impl_bcast!([f32; M], [1], [0], { M });

// 1d -> 3d
impl_bcast!([f32; M], [2], [0, 1], { M });
impl_bcast!([f32; M], [1], [0, 2], { M });
impl_bcast!([f32; M], [0], [1, 2], { M });

// 1d -> 4d
impl_bcast!([f32; M], [3], [0, 1, 2], { M });
impl_bcast!([f32; M], [2], [0, 1, 3], { M });
impl_bcast!([f32; M], [1], [0, 2, 3], { M });
impl_bcast!([f32; M], [0], [1, 2, 3], { M });

// 2d -> 3d
impl_bcast!([[f32; N]; M], [0, 1], [-1], {M, N});
impl_bcast!([[f32; N]; M], [0, 1], [2], {M, N});
impl_bcast!([[f32; N]; M], [0, 2], [1], {M, N});
impl_bcast!([[f32; N]; M], [1, 2], [0], {M, N});

// 2d -> 4d
impl_bcast!([[f32; N]; M], [2, 3], [0, 1], {M, N});
impl_bcast!([[f32; N]; M], [1, 3], [0, 2], {M, N});
impl_bcast!([[f32; N]; M], [1, 2], [0, 3], {M, N});
impl_bcast!([[f32; N]; M], [0, 3], [1, 2], {M, N});
impl_bcast!([[f32; N]; M], [0, 2], [1, 3], {M, N});
impl_bcast!([[f32; N]; M], [0, 1], [2, 3], {M, N});

// 3d -> 4d
impl_bcast!([[[f32; O]; N]; M], [0, 1, 2], [-1], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [0, 1, 2], [3], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [0, 1, 3], [2], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [0, 2, 3], [1], {M, N, O});
impl_bcast!([[[f32; O]; N]; M], [1, 2, 3], [0], {M, N, O});
