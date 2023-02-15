use super::*;

/// Marker for shapes that can be reduced to [Shape] `S` along [Axes] `Ax`.
pub trait ReduceShapeTo<S, Ax>: Sized {}

/// Marker for shapes that can be broadcasted to [Shape] `S` along [Axes] `Ax`.
pub trait BroadcastShapeTo<S, Ax>: Sized {}

/// Marker for shapes that can have their [Axes] `Ax` reduced. See Self::Reduced
/// for the resulting type.
pub trait ReduceShape<Ax>: Sized + Shape + HasAxes<Ax> + ReduceShapeTo<Self::Reduced, Ax> {
    type Reduced: Shape + BroadcastShapeTo<Self, Ax>;
}

impl ReduceShapeTo<(), Axis<0>> for () {}
impl ReduceShape<Axis<0>> for () {
    type Reduced = ();
}
impl<Src: Shape, Dst: Shape + ReduceShapeTo<Src, Ax>, Ax> BroadcastShapeTo<Dst, Ax> for Src {}

macro_rules! broadcast_to_array {
    ($SrcNum:literal, (), $DstNum:literal, ($($DstDims:tt),*), $Axes:ty) => {
        impl ReduceShapeTo<(), $Axes> for [usize; $DstNum] {}
        impl ReduceShape<$Axes> for [usize; $DstNum] {
            type Reduced = ();
        }
    };
    ($SrcNum:literal, ($($SrcDims:tt),*), $DstNum:literal, ($($DstDims:tt),*), $Axes:ty) => {
        impl ReduceShapeTo<[usize; $SrcNum], $Axes> for [usize; $DstNum] {}
        impl ReduceShape<$Axes> for [usize; $DstNum] {
            type Reduced = [usize; $SrcNum];
        }
    };
}

macro_rules! broadcast_to {
    ($SrcNum:literal, ($($SrcDims:tt),*), $DstNum:literal, ($($DstDims:tt),*), $Axes:ty) => {
        impl<$($DstDims: Dim, )*> ReduceShapeTo<($($SrcDims, )*), $Axes> for ($($DstDims, )*) {}
        impl<$($DstDims: Dim, )*> ReduceShape<$Axes> for ($($DstDims, )*) {
            type Reduced = ($($SrcDims, )*);
        }
        broadcast_to_array!($SrcNum, ($($SrcDims),*), $DstNum, ($($DstDims),*), $Axes);
    };
}
broadcast_to!(0, (), 1, (M), Axis<0>);
broadcast_to!(0, (), 2, (M, N), Axes2<0, 1>);
broadcast_to!(0, (), 3, (M, N, O), Axes3<0, 1, 2>);
broadcast_to!(0, (), 4, (M, N, O, P), Axes4<0, 1, 2, 3>);
broadcast_to!(0, (), 5, (M, N, O, P, Q), Axes5<0, 1, 2, 3, 4>);
broadcast_to!(0, (), 6, (M, N, O, P, Q, R), Axes6<0, 1, 2, 3, 4, 5>);

broadcast_to!(1, (M), 2, (M, N), Axis<1>);
broadcast_to!(1, (N), 2, (M, N), Axis<0>);
broadcast_to!(1, (M), 3, (M, N, O), Axes2<1, 2>);
broadcast_to!(1, (N), 3, (M, N, O), Axes2<0, 2>);
broadcast_to!(1, (O), 3, (M, N, O), Axes2<0, 1>);
broadcast_to!(1, (M), 4, (M, N, O, P), Axes3<1, 2, 3>);
broadcast_to!(1, (N), 4, (M, N, O, P), Axes3<0, 2, 3>);
broadcast_to!(1, (O), 4, (M, N, O, P), Axes3<0, 1, 3>);
broadcast_to!(1, (P), 4, (M, N, O, P), Axes3<0, 1, 2>);

broadcast_to!(2, (M, N), 3, (M, N, O), Axis<2>);
broadcast_to!(2, (M, O), 3, (M, N, O), Axis<1>);
broadcast_to!(2, (N, O), 3, (M, N, O), Axis<0>);
broadcast_to!(2, (M, N), 4, (M, N, O, P), Axes2<2, 3>);
broadcast_to!(2, (M, O), 4, (M, N, O, P), Axes2<1, 3>);
broadcast_to!(2, (N, O), 4, (M, N, O, P), Axes2<0, 3>);
broadcast_to!(2, (M, P), 4, (M, N, O, P), Axes2<1, 2>);
broadcast_to!(2, (N, P), 4, (M, N, O, P), Axes2<0, 2>);
broadcast_to!(2, (O, P), 4, (M, N, O, P), Axes2<0, 1>);

broadcast_to!(3, (M, N, O), 4, (M, N, O, P), Axis<3>);
broadcast_to!(3, (M, N, P), 4, (M, N, O, P), Axis<2>);
broadcast_to!(3, (M, O, P), 4, (M, N, O, P), Axis<1>);
broadcast_to!(3, (N, O, P), 4, (M, N, O, P), Axis<0>);

/// Internal implementation for broadcasting strides
pub trait BroadcastStridesTo<S: Shape, Ax>: Shape + BroadcastShapeTo<S, Ax> {
    fn check(&self, dst: &S);
    fn broadcast_strides(&self, strides: Self::Concrete) -> S::Concrete;
}

impl<Src: Shape, Dst: Shape, Ax: Axes> BroadcastStridesTo<Dst, Ax> for Src
where
    Self: BroadcastShapeTo<Dst, Ax>,
{
    #[inline(always)]
    fn check(&self, dst: &Dst) {
        let src_dims = self.concrete();
        let dst_dims = dst.concrete();
        let mut j = 0;
        for i in 0..Dst::NUM_DIMS {
            if !Ax::as_array().into_iter().any(|x| x == i as isize) {
                assert_eq!(dst_dims[i], src_dims[j]);
                j += 1;
            }
        }
    }

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

/// Internal implementation for reducing a shape
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
    fn test_check() {
        BroadcastStridesTo::<(usize, usize), Axis<1>>::check(&(1,), &(1, 2));
    }

    #[test]
    #[should_panic]
    fn test_check_failures() {
        BroadcastStridesTo::<(usize, usize), Axis<1>>::check(&(1,), &(2, 2));
    }

    #[test]
    fn test_no_conflict_reductions() {
        let src = (1, Const::<2>, 3, Const::<4>);

        let dst: (usize, Const<2>) = src.reduced();
        assert_eq!(dst, (1, Const::<2>));

        let dst: (Const<2>, usize) = src.reduced();
        assert_eq!(dst, (Const::<2>, 3));

        let dst: (usize, usize) = src.reduced();
        assert_eq!(dst, (1, 3));
    }

    #[test]
    fn test_conflicting_reductions() {
        let src = (1, 2, Const::<3>);

        let dst = ReduceStridesTo::<_, Axis<1>>::reduced(&src);
        assert_eq!(dst, (1, Const::<3>));

        let dst = ReduceStridesTo::<_, Axis<0>>::reduced(&src);
        assert_eq!(dst, (2, Const::<3>));
    }

    #[test]
    fn test_broadcast_strides() {
        let src = (1,);
        let dst_strides =
            BroadcastStridesTo::<(usize, usize, usize), Axes2<0, 2>>::broadcast_strides(
                &src,
                src.strides(),
            );
        assert_eq!(dst_strides, [0, 1, 0]);
    }
}
