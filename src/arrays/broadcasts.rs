use super::*;

pub trait ReduceShapeTo<S, Ax>: Sized {}
pub trait BroadcastShapeTo<S, Ax>: Sized {}

pub trait ReduceShape<Ax>: Sized + Shape + HasAxes<Ax> + ReduceShapeTo<Self::Reduced, Ax> {
    type Reduced: Shape + Default + BroadcastShapeTo<Self, Ax>;
}

impl ReduceShapeTo<(), Axis<0>> for () {}
impl ReduceShape<Axis<0>> for () {
    type Reduced = ();
}
impl<Src: Shape, Dst: Shape + ReduceShapeTo<Src, Ax>, Ax> BroadcastShapeTo<Dst, Ax> for Src {}

macro_rules! broadcast_to {
    (($($SrcDims:tt),*), ($($DstDims:tt),*), $Axes:ty) => {
impl<$($DstDims: Dim, )*> ReduceShapeTo<($($SrcDims, )*), $Axes> for ($($DstDims, )*) {}
impl<$($DstDims: Dim + Default, )*> ReduceShape<$Axes> for ($($DstDims, )*) {
    type Reduced = ($($SrcDims, )*);
}
    };
}
broadcast_to!((), (M), Axis<0>);
broadcast_to!((), (M, N), Axes2<0, 1>);
broadcast_to!((), (M, N, O), Axes3<0, 1, 2>);
broadcast_to!((), (M, N, O, P), Axes4<0, 1, 2, 3>);
broadcast_to!((), (M, N, O, P, Q), Axes5<0, 1, 2, 3, 4>);
broadcast_to!((), (M, N, O, P, Q, R), Axes6<0, 1, 2, 3, 4, 5>);

broadcast_to!((M), (M, N), Axis<1>);
broadcast_to!((N), (M, N), Axis<0>);
broadcast_to!((M), (M, N, O), Axes2<1, 2>);
broadcast_to!((N), (M, N, O), Axes2<0, 2>);
broadcast_to!((O), (M, N, O), Axes2<0, 1>);
broadcast_to!((M), (M, N, O, P), Axes3<1, 2, 3>);
broadcast_to!((N), (M, N, O, P), Axes3<0, 2, 3>);
broadcast_to!((O), (M, N, O, P), Axes3<0, 1, 3>);
broadcast_to!((P), (M, N, O, P), Axes3<0, 1, 2>);

broadcast_to!((M, N), (M, N, O), Axis<2>);
broadcast_to!((M, O), (M, N, O), Axis<1>);
broadcast_to!((N, O), (M, N, O), Axis<0>);
broadcast_to!((M, N), (M, N, O, P), Axes2<2, 3>);
broadcast_to!((M, O), (M, N, O, P), Axes2<1, 3>);
broadcast_to!((N, O), (M, N, O, P), Axes2<0, 3>);
broadcast_to!((M, P), (M, N, O, P), Axes2<1, 2>);
broadcast_to!((N, P), (M, N, O, P), Axes2<0, 2>);
broadcast_to!((O, P), (M, N, O, P), Axes2<0, 1>);

broadcast_to!((M, N, O), (M, N, O, P), Axis<3>);
broadcast_to!((M, N, P), (M, N, O, P), Axis<2>);
broadcast_to!((M, O, P), (M, N, O, P), Axis<1>);
broadcast_to!((N, O, P), (M, N, O, P), Axis<0>);

pub trait BroadcastStridesTo<S: Shape, Axes>: Shape + BroadcastShapeTo<S, Axes> {
    fn broadcast_strides(&self, strides: Self::Concrete) -> S::Concrete;
}

impl<Src: Shape, Dst: Shape, Ax: Axes> BroadcastStridesTo<Dst, Ax> for Src
where
    Self: BroadcastShapeTo<Dst, Ax>,
{
    #[inline(always)]
    fn broadcast_strides(&self, strides: Self::Concrete) -> Dst::Concrete {
        let mut new_strides: Dst::Concrete = Default::default();
        let mut j = 0;
        for i in 0..Dst::NUM_DIMS {
            if !Ax::as_array().into_iter().any(|x| x == i as isize) {
                new_strides[i] = strides[j];
                j += 1;
            }
        }
        new_strides
    }
}
