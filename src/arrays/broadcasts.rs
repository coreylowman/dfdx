use super::*;

pub trait BroadcastShapeTo<S, Ax>: Sized {}

pub trait ReduceShapeTo<S: BroadcastShapeTo<Self, Ax>, Ax>: Shape + Sized {}
impl<Src: Shape, Dst: Shape + BroadcastShapeTo<Src, Ax>, Ax> ReduceShapeTo<Dst, Ax> for Src {}

pub trait ReduceShape<Ax>: Shape + HasAxes<Ax> {
    type Reduced: Shape + Default + BroadcastStridesTo<Self, Ax>;
}

macro_rules! broadcast_to {
    ($SrcTy:ty, $DstTy:ty, $Axes:ty, {$($ConstVars:tt),*}) => {
impl<$(const $ConstVars: usize, )*> BroadcastShapeTo<$DstTy, $Axes> for $SrcTy {}
impl<$(const $ConstVars: usize, )*> ReduceShape<$Axes> for $DstTy {
    type Reduced = $SrcTy;
}
    };
}
broadcast_to!(Rank0, Rank1<M>, Axis<0>, { M });
broadcast_to!(Rank0, Rank2<M, N>, Axes2<0, 1>, { M, N });
broadcast_to!(Rank0, Rank3<M, N, O>, Axes3<0, 1, 2>, { M, N, O });
broadcast_to!(Rank0, Rank4<M, N, O, P>, Axes4<0, 1, 2, 3>, { M, N, O, P });

broadcast_to!(Rank1<M>, Rank2<M, N>, Axis<1>, { M, N });
broadcast_to!(Rank1<N>, Rank2<M, N>, Axis<0>, { M, N });
broadcast_to!(Rank1<M>, Rank3<M, N, O>, Axes2<1, 2>, { M, N, O });
broadcast_to!(Rank1<N>, Rank3<M, N, O>, Axes2<0, 2>, { M, N, O });
broadcast_to!(Rank1<O>, Rank3<M, N, O>, Axes2<0, 1>, { M, N, O });
broadcast_to!(Rank1<M>, Rank4<M, N, O, P>, Axes3<1, 2, 3>, { M, N, O, P });
broadcast_to!(Rank1<N>, Rank4<M, N, O, P>, Axes3<0, 2, 3>, { M, N, O, P });
broadcast_to!(Rank1<O>, Rank4<M, N, O, P>, Axes3<0, 1, 3>, { M, N, O, P });
broadcast_to!(Rank1<P>, Rank4<M, N, O, P>, Axes3<0, 1, 2>, { M, N, O, P });

broadcast_to!(Rank2<M, N>, Rank3<M, N, O>, Axis<2>, { M, N, O });
broadcast_to!(Rank2<M, O>, Rank3<M, N, O>, Axis<1>, { M, N, O });
broadcast_to!(Rank2<N, O>, Rank3<M, N, O>, Axis<0>, { M, N, O });
broadcast_to!(Rank2<M, N>, Rank4<M, N, O, P>, Axes2<2, 3>, { M, N, O, P });
broadcast_to!(Rank2<M, O>, Rank4<M, N, O, P>, Axes2<1, 3>, { M, N, O, P });
broadcast_to!(Rank2<N, O>, Rank4<M, N, O, P>, Axes2<0, 3>, { M, N, O, P });
broadcast_to!(Rank2<M, P>, Rank4<M, N, O, P>, Axes2<1, 2>, { M, N, O, P });
broadcast_to!(Rank2<N, P>, Rank4<M, N, O, P>, Axes2<0, 2>, { M, N, O, P });
broadcast_to!(Rank2<O, P>, Rank4<M, N, O, P>, Axes2<0, 1>, { M, N, O, P });

broadcast_to!(Rank3<M, N, O>, Rank4<M, N, O, P>, Axis<3>, {M, N, O, P});
broadcast_to!(Rank3<M, N, P>, Rank4<M, N, O, P>, Axis<2>, {M, N, O, P});
broadcast_to!(Rank3<M, O, P>, Rank4<M, N, O, P>, Axis<1>, {M, N, O, P});
broadcast_to!(Rank3<N, O, P>, Rank4<M, N, O, P>, Axis<0>, {M, N, O, P});

pub trait BroadcastStridesTo<S: Shape, Axes>: Shape + BroadcastShapeTo<S, Axes> {
    fn broadcast_strides(&self, strides: StridesFor<Self>) -> StridesFor<S>;
}

impl<Src: Shape, Dst: Shape, Ax: Axes> BroadcastStridesTo<Dst, Ax> for Src
where
    Self: BroadcastShapeTo<Dst, Ax>,
{
    #[inline(always)]
    fn broadcast_strides(&self, strides: StridesFor<Self>) -> StridesFor<Dst> {
        let mut new_strides: Dst::Concrete = Default::default();
        let mut j = 0;
        for i in 0..Dst::NUM_DIMS {
            if !Ax::as_array().into_iter().any(|x| x == i as isize) {
                new_strides[i] = strides.0[j];
                j += 1;
            }
        }
        StridesFor(new_strides)
    }
}
