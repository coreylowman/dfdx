use super::shape::{Dim, Shape};

pub trait ReplaceDimTo<Dst: Shape, const I: isize>: Shape {
    type Index: Shape;
    #[inline]
    fn replace(&self, idx: Self::Index) -> Dst {
        if Dst::NUM_DIMS == Self::NUM_DIMS {
            // replace case
            let src_dims = self.concrete();
            let mut dst_dims: Dst::Concrete = Default::default();
            for i in 0..Dst::NUM_DIMS {
                dst_dims[i] = src_dims[i];
            }
            dst_dims[I as usize] = idx.concrete().into_iter().last().unwrap();
            Dst::from_concrete(&dst_dims).unwrap()
        } else {
            // remove case. this is similar to ReduceShapeTo::reduce
            let src_dims = self.concrete();
            let mut dst_dims: Dst::Concrete = Default::default();
            let mut i_dst = 0;
            for i_src in 0..Self::NUM_DIMS {
                if i_src != I as usize {
                    dst_dims[i_dst] = src_dims[i_src];
                    i_dst += 1;
                }
            }
            Dst::from_concrete(&dst_dims).unwrap()
        }
    }
}

macro_rules! replace {
    (($($DimVars:tt),*), $Ax:tt, $Dst:ty, $Idx:ty) => {
impl<$($DimVars: Dim, )* New: Dim> ReplaceDimTo<$Dst, $Ax> for ($($DimVars, )*) {
    type Index = $Idx;
}
    };
}

macro_rules! removed {
    (($($DimVars:tt),*), $Ax:tt, $Dst:ty, $Idx:ty) => {
impl<$($DimVars: Dim, )*> ReplaceDimTo<$Dst, $Ax> for ($($DimVars, )*) {
    type Index = $Idx;
}
    };
}

replace!((D1), 0, (New,), (New,));
removed!((D1), 0, (), ());

replace!((D1, D2), 0, (New, D2), (New,));
removed!((D1, D2), 0, (D2,), ());
replace!((D1, D2), 1, (D1, New), (D1, New,));
removed!((D1, D2), 1, (D1,), (D1,));

replace!((D1, D2, D3), 0, (New, D2, D3), (New,));
removed!((D1, D2, D3), 0, (D2, D3), ());
replace!((D1, D2, D3), 1, (D1, New, D3), (D1, New,));
removed!((D1, D2, D3), 1, (D1, D3), (D1,));
replace!((D1, D2, D3), 2, (D1, D2, New), (D1, D2, New));
removed!((D1, D2, D3), 2, (D1, D2,), (D1, D2));

replace!((D1, D2, D3, D4), 0, (New, D2, D3, D4), (New,));
removed!((D1, D2, D3, D4), 0, (D2, D3, D4), ());
replace!((D1, D2, D3, D4), 1, (D1, New, D3, D4), (D1, New));
removed!((D1, D2, D3, D4), 1, (D1, D3, D4), (D1,));
replace!((D1, D2, D3, D4), 2, (D1, D2, New, D4), (D1, D2, New));
removed!((D1, D2, D3, D4), 2, (D1, D2, D4), (D1, D2));
replace!((D1, D2, D3, D4), 3, (D1, D2, D3, New), (D1, D2, D3, New));
removed!((D1, D2, D3, D4), 3, (D1, D2, D3), (D1, D2, D3));
