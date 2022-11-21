use crate::arrays::C;
use crate::{Assert, ConstTrue};

pub trait HasSameNumelAs<Dst> {}

macro_rules! impl_same_num_elements {
    ([$($SrcVs:tt),*], $SrcNumEl:tt, [$($DstVs:tt),*], $DstNumEl:tt) => {
impl<$(const $SrcVs: usize, )* $(const $DstVs: usize, )*> HasSameNumelAs<($(C<$SrcVs>, )*)> for ($(C<$DstVs>, )*)
where
    Assert<{ $SrcNumEl == $DstNumEl }>: ConstTrue {}
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
