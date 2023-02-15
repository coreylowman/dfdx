use super::{
    axes::{Axes, Axis},
    shape::{Dim, Shape},
};

/// Marker for shapes that can be indexed and have a dimension removed
pub trait RemoveDimTo<Dst: Shape, Idx: Shape>: Shape {
    type Ax: Axes<Array = [isize; 1]>;

    /// All dimensions of idx should be the same as the dimensions of Self
    #[inline]
    fn check(&self, idx: &Idx) {
        assert!(Idx::NUM_DIMS <= Self::NUM_DIMS);
        let src_dims = self.concrete();
        let idx_dims = idx.concrete();
        for i in 0..Idx::NUM_DIMS {
            assert_eq!(src_dims[i], idx_dims[i]);
        }
    }

    #[inline]
    fn remove(&self, _: Idx) -> Dst {
        let ax = Self::Ax::as_array()[0] as usize;
        // remove case. this is similar to ReduceShapeTo::reduce
        let src_dims = self.concrete();
        let mut dst_dims: Dst::Concrete = Default::default();
        let mut i_dst = 0;
        for i_src in 0..Self::NUM_DIMS {
            if i_src != ax {
                dst_dims[i_dst] = src_dims[i_src];
                i_dst += 1;
            }
        }
        Dst::from_concrete(&dst_dims).unwrap()
    }
}

/// Marker for shapes that can be indexed and have a dimension replaced with a new one
pub trait ReplaceDimTo<Dst: Shape, Idx: Shape>: Shape {
    type Ax: Axes<Array = [isize; 1]>;

    /// All dimensions of idx *up to last dimension* (which is new)
    /// should be the same as the dimensions of Self
    #[inline]
    fn check(&self, idx: &Idx) {
        if Self::NUM_DIMS == Dst::NUM_DIMS {
            // replace 1 dim case
            assert!(Idx::NUM_DIMS <= Self::NUM_DIMS);
            let src_dims = self.concrete();
            let idx_dims = idx.concrete();
            for i in 0..Idx::NUM_DIMS - 1 {
                assert_eq!(src_dims[i], idx_dims[i]);
            }
        } else {
            // batch replace case - we actually don't need to check this case
            // at all
        }
    }

    #[inline]
    fn replace(&self, idx: Idx) -> Dst {
        let ax = Self::Ax::as_array()[0] as usize;
        if Self::NUM_DIMS == Dst::NUM_DIMS {
            // replace 1 dim case
            let src_dims = self.concrete();
            let mut dst_dims: Dst::Concrete = Default::default();
            for i in 0..Dst::NUM_DIMS {
                dst_dims[i] = src_dims[i];
            }
            dst_dims[ax] = idx.concrete().into_iter().last().unwrap();
            Dst::from_concrete(&dst_dims).unwrap()
        } else {
            // batch replace case (M, N) * (B, Z) -> (B, Z, N)
            assert_eq!(Dst::NUM_DIMS, Self::NUM_DIMS + 1);
            assert_eq!(Self::NUM_DIMS, <Idx as Shape>::NUM_DIMS);
            assert_eq!(ax, 0);
            let src_dims = self.concrete();
            let idx_dims = idx.concrete();
            let mut dst_dims: Dst::Concrete = Default::default();
            for i in 0..Dst::NUM_DIMS {
                dst_dims[i] = if i < Self::NUM_DIMS {
                    idx_dims[i]
                } else {
                    src_dims[i - 1]
                };
            }
            Dst::from_concrete(&dst_dims).unwrap()
        }
    }
}

macro_rules! replace {
    (($($DimVars:tt),*), $Ax:ty, $Dst:ty, $Idx:ty) => {
impl<$($DimVars: Dim, )* New: Dim> ReplaceDimTo<$Dst, $Idx> for ($($DimVars, )*) {
    type Ax = $Ax;
}
    };
}

macro_rules! removed {
    (($($DimVars:tt),*), $Ax:ty, $Dst:ty, $Idx:ty) => {
impl<$($DimVars: Dim, )*> RemoveDimTo<$Dst, $Idx> for ($($DimVars, )*) {
    type Ax = $Ax;
}
    };
}

replace!((D1), Axis<0>, (New,), (New,));
removed!((D1), Axis<0>, (), ());

replace!((D1, D2), Axis<0>, (New, D2), (New,));
removed!((D1, D2), Axis<0>, (D2,), ());
replace!((D1, D2), Axis<1>, (D1, New), (D1, New,));
removed!((D1, D2), Axis<1>, (D1,), (D1,));

replace!((D1, D2, D3), Axis<0>, (New, D2, D3), (New,));
removed!((D1, D2, D3), Axis<0>, (D2, D3), ());
replace!((D1, D2, D3), Axis<1>, (D1, New, D3), (D1, New,));
removed!((D1, D2, D3), Axis<1>, (D1, D3), (D1,));
replace!((D1, D2, D3), Axis<2>, (D1, D2, New), (D1, D2, New));
removed!((D1, D2, D3), Axis<2>, (D1, D2,), (D1, D2));

replace!((D1, D2, D3, D4), Axis<0>, (New, D2, D3, D4), (New,));
removed!((D1, D2, D3, D4), Axis<0>, (D2, D3, D4), ());
replace!((D1, D2, D3, D4), Axis<1>, (D1, New, D3, D4), (D1, New));
removed!((D1, D2, D3, D4), Axis<1>, (D1, D3, D4), (D1,));
replace!((D1, D2, D3, D4), Axis<2>, (D1, D2, New, D4), (D1, D2, New));
removed!((D1, D2, D3, D4), Axis<2>, (D1, D2, D4), (D1, D2));
replace!(
    (D1, D2, D3, D4),
    Axis<3>,
    (D1, D2, D3, New),
    (D1, D2, D3, New)
);
removed!((D1, D2, D3, D4), Axis<3>, (D1, D2, D3), (D1, D2, D3));

// batched select
impl<Batch: Dim, Seq: Dim, S1: Dim, S2: Dim> ReplaceDimTo<(Batch, Seq, S2), (Batch, Seq)>
    for (S1, S2)
{
    type Ax = Axis<0>;
}
