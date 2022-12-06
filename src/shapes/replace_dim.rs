use super::{
    axes::{Axes, Axis},
    shape::{Dim, Shape},
};

pub trait ReplaceDimTo<Dst: Shape, Ax: Axes<Array = [isize; 1]>>: Shape {
    type Index: Shape;
    #[inline]
    fn replace(&self, idx: Self::Index) -> Dst {
        let ax = Ax::as_array()[0] as usize;
        if Dst::NUM_DIMS == Self::NUM_DIMS {
            // replace case
            let src_dims = self.concrete();
            let mut dst_dims: Dst::Concrete = Default::default();
            for i in 0..Dst::NUM_DIMS {
                dst_dims[i] = src_dims[i];
            }
            dst_dims[ax] = idx.concrete().into_iter().last().unwrap();
            Dst::from_concrete(&dst_dims).unwrap()
        } else if Dst::NUM_DIMS == Self::NUM_DIMS - 1 {
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
        } else {
            // batch select case (M, N) * (B, Z) -> (B, Z, N)
            assert_eq!(Dst::NUM_DIMS, Self::NUM_DIMS + 1);
            assert_eq!(Self::NUM_DIMS, <Self::Index as Shape>::NUM_DIMS);
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
impl<$($DimVars: Dim, )* New: Dim> ReplaceDimTo<$Dst, $Ax> for ($($DimVars, )*) {
    type Index = $Idx;
}
    };
}

macro_rules! removed {
    (($($DimVars:tt),*), $Ax:ty, $Dst:ty, $Idx:ty) => {
impl<$($DimVars: Dim, )*> ReplaceDimTo<$Dst, $Ax> for ($($DimVars, )*) {
    type Index = $Idx;
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
impl<Batch: Dim, Seq: Dim, S1: Dim, S2: Dim> ReplaceDimTo<(Batch, Seq, S2), Axis<0>> for (S1, S2) {
    type Index = (Batch, Seq);
}
