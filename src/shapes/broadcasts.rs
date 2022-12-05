use super::*;

pub trait ReduceShapeTo<S, Ax>: Sized {}
pub trait BroadcastShapeTo<S, Ax>: Sized {}

pub trait ReduceShape<Ax>: Sized + Shape + HasAxes<Ax> + ReduceShapeTo<Self::Reduced, Ax> {
    type Reduced: Shape + BroadcastShapeTo<Self, Ax>;
}

impl ReduceShapeTo<(), Axis<0>> for () {}
impl ReduceShape<Axis<0>> for () {
    type Reduced = ();
}
impl<Src: Shape, Dst: Shape + ReduceShapeTo<Src, Ax>, Ax> BroadcastShapeTo<Dst, Ax> for Src {}

macro_rules! broadcast_to {
    (($($SrcDims:tt),*), ($($DstDims:tt),*), $Axes:ty) => {
impl<$($DstDims: Dim, )*> ReduceShapeTo<($($SrcDims, )*), $Axes> for ($($DstDims, )*) {}
impl<$($DstDims: Dim, )*> ReduceShape<$Axes> for ($($DstDims, )*) {
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

pub trait BroadcastStridesTo<S: Shape, Ax>: Shape + BroadcastShapeTo<S, Ax> {
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

pub trait ReduceStridesTo<S: Shape, Ax>: Shape + ReduceShapeTo<S, Ax> {
    fn reduced(&self) -> S;
}

impl<Src: Shape, Dst: Shape, Ax: Axes> ReduceStridesTo<Dst, Ax> for Src
where
    Self: ReduceShapeTo<Dst, Ax>,
{
    #[inline(always)]
    fn reduced(&self) -> Dst {
        let src_dims = self.concrete();
        let mut dst_dims: Dst::Concrete = Default::default();
        let mut i_dst = 0;
        for i_src in 0..Src::NUM_DIMS {
            if !Ax::as_array().into_iter().any(|x| x == i_src as isize) {
                dst_dims[i_dst] = src_dims[i_src];
                i_dst += 1;
            }
        }
        Dst::from_concrete(&dst_dims).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_conflict_reductions() {
        let src = (Dyn(1), Const::<2>, Dyn(3), Const::<4>);

        let dst: (Dyn, Const<2>) = src.reduced();
        assert_eq!(dst, (Dyn(1), Const::<2>));

        let dst: (Const<2>, Dyn) = src.reduced();
        assert_eq!(dst, (Const::<2>, Dyn(3)));

        let dst: (Dyn, Dyn) = src.reduced();
        assert_eq!(dst, (Dyn(1), Dyn(3)));
    }

    #[test]
    fn test_conflicting_reductions() {
        let src = (Dyn(1), Dyn(2), Const::<3>);

        let dst = ReduceStridesTo::<_, Axis<1>>::reduced(&src);
        assert_eq!(dst, (Dyn(1), Const::<3>));

        let dst = ReduceStridesTo::<_, Axis<0>>::reduced(&src);
        assert_eq!(dst, (Dyn(2), Const::<3>));
    }

    #[test]
    fn test_broadcast_strides() {
        let src = (Dyn(1),);
        let dst_strides = BroadcastStridesTo::<(Dyn, Dyn, Dyn), Axes2<0, 2>>::broadcast_strides(
            &src,
            src.strides(),
        );
        assert_eq!(dst_strides, [0, 1, 0]);
    }
}
