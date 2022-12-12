#![allow(unused_imports)]

use crate::shapes::Const;
use crate::{Assert, ConstTrue};

/// Marker for shapes that have the same number of elements as `Dst`
pub trait HasSameNumelAs<Dst> {}

macro_rules! impl_same_num_elements {
    ([$($SrcVs:tt),*], $SrcNumEl:tt, [$($DstVs:tt),*], $DstNumEl:tt) => {
#[cfg(feature = "nightly")]
impl<$(const $SrcVs: usize, )* $(const $DstVs: usize, )*> HasSameNumelAs<($(Const<$SrcVs>, )*)> for ($(Const<$DstVs>, )*)
where
    Assert<{ $DstNumEl == $SrcNumEl }>: ConstTrue {}
    };
}

macro_rules! impl_for {
    ([$($SrcVs:tt),*], $SrcNumEl:tt) => {
        impl_same_num_elements!([$($SrcVs),*], $SrcNumEl, [], (1));
        impl_same_num_elements!([$($SrcVs),*], $SrcNumEl, [M], (M));
        impl_same_num_elements!([$($SrcVs),*], $SrcNumEl, [M, N], (M * N));
        impl_same_num_elements!([$($SrcVs),*], $SrcNumEl, [M, N, O], (M * N * O));
        impl_same_num_elements!([$($SrcVs),*], $SrcNumEl, [M, N, O, P], (M * N * O * P));
    };
}

impl_for!([], (1));
impl_for!([S], (S));
impl_for!([S, T], (S * T));
impl_for!([S, T, U], (S * T * U));
impl_for!([S, T, U, V], (S * T * U * V));
