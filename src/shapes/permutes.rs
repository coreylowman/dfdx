use super::*;

/// Marker for shapes that can be permuted into `Dst` by using `Ax`
/// as the new indices.
///
/// E.g. `PermuteShapeTo<_, Axes2<1, 0>>` would mean you can reverse
/// axes 0 and 1.
pub trait PermuteShapeTo<Dst, Ax> {}

pub trait PermuteStridesTo<S: Shape, Ax>: Shape + PermuteShapeTo<S, Ax> {
    fn permuted(&self) -> S;
    fn permute_strides(&self, strides: Self::Concrete) -> S::Concrete;
}

impl<Src: Shape, Dst: Shape, Ax: Axes> PermuteStridesTo<Dst, Ax> for Src
where
    Self: PermuteShapeTo<Dst, Ax>,
{
    #[inline(always)]
    fn permuted(&self) -> Dst {
        let src_dims = self.concrete();
        let mut dst_dims: Dst::Concrete = Default::default();
        for (i_dst, i_src) in Ax::as_array().into_iter().enumerate() {
            dst_dims[i_dst] = src_dims[i_src as usize];
        }
        Dst::from_concrete(&dst_dims).unwrap()
    }

    #[inline(always)]
    fn permute_strides(&self, src_strides: Self::Concrete) -> Dst::Concrete {
        let mut dst_strides: Dst::Concrete = Default::default();
        for (i, idx) in Ax::as_array().into_iter().enumerate() {
            dst_strides[i] = src_strides[idx as usize];
        }
        dst_strides
    }
}

#[rustfmt::skip]
macro_rules! d { (0) => { D1 }; (1) => { D2 }; (2) => { D3 }; (3) => { D4 }; (4) => { D5 }; (5) => { D6 }; }

macro_rules! impl_permute {
    ($Ax0:tt, $Ax1:tt) => {
        impl<D1: Dim, D2: Dim> PermuteShapeTo<(d!($Ax0), d!($Ax1)), Axes2<$Ax0, $Ax1>>
            for (D1, D2)
        {
        }
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt) => {
        impl<D1: Dim, D2: Dim, D3: Dim>
            PermuteShapeTo<(d!($Ax0), d!($Ax1), d!($Ax2)), Axes3<$Ax0, $Ax1, $Ax2>>
            for (D1, D2, D3)
        {
        }
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt) => {
        impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim>
            PermuteShapeTo<(d!($Ax0), d!($Ax1), d!($Ax2), d!($Ax3)), Axes4<$Ax0, $Ax1, $Ax2, $Ax3>>
            for (D1, D2, D3, D4)
        {
        }
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt) => {
        impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim>
            PermuteShapeTo<
                (d!($Ax0), d!($Ax1), d!($Ax2), d!($Ax3), d!($Ax4)),
                Axes5<$Ax0, $Ax1, $Ax2, $Ax3, $Ax4>,
            > for (D1, D2, D3, D4, D5)
        {
        }
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt) => {
        impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim>
            PermuteShapeTo<
                (d!($Ax0), d!($Ax1), d!($Ax2), d!($Ax3), d!($Ax4), d!($Ax5)),
                Axes6<$Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5>,
            > for (D1, D2, D3, D4, D5, D6)
        {
        }
    };
}

/// Expand out all the possible permutations for 2-4d
macro_rules! permutations {
    ([$Ax0:tt, $Ax1:tt]) => {
        impl_permute!($Ax1, $Ax0);
    };
    ([$Ax0:tt, $Ax1:tt, $Ax2:tt]) => {
        impl_permute!($Ax0, $Ax2, $Ax1);
        impl_permute!($Ax1, $Ax0, $Ax2);
        impl_permute!($Ax1, $Ax2, $Ax0);
        impl_permute!($Ax2, $Ax0, $Ax1);
        impl_permute!($Ax2, $Ax1, $Ax0);
    };

    ([$Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt]) => {
        permutations!($Ax0, [$Ax1, $Ax2, $Ax3]);
        permutations!($Ax1, [$Ax0, $Ax2, $Ax3]);
        permutations!($Ax2, [$Ax0, $Ax1, $Ax3]);
        permutations!($Ax3, [$Ax0, $Ax1, $Ax2]);
    };
    ($Ax0:tt, [$Ax1:tt, $Ax2:tt, $Ax3:tt]) => {
        permutations!($Ax0, $Ax1, [$Ax2, $Ax3]);
        permutations!($Ax0, $Ax2, [$Ax1, $Ax3]);
        permutations!($Ax0, $Ax3, [$Ax1, $Ax2]);
    };
    ($Ax0:tt, $Ax1:tt, [$Ax2:tt, $Ax3:tt]) => {
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax3);
        impl_permute!($Ax0, $Ax1, $Ax3, $Ax2);
    };

    ([$Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt]) => {
        permutations!($Ax0, [$Ax1, $Ax2, $Ax3, $Ax4]);
        permutations!($Ax1, [$Ax0, $Ax2, $Ax3, $Ax4]);
        permutations!($Ax2, [$Ax0, $Ax1, $Ax3, $Ax4]);
        permutations!($Ax3, [$Ax0, $Ax1, $Ax2, $Ax4]);
        permutations!($Ax4, [$Ax0, $Ax1, $Ax2, $Ax3]);
    };
    ($Ax0:tt, [$Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt]) => {
        permutations!($Ax0, $Ax1, [$Ax2, $Ax3, $Ax4]);
        permutations!($Ax0, $Ax2, [$Ax1, $Ax3, $Ax4]);
        permutations!($Ax0, $Ax3, [$Ax1, $Ax2, $Ax4]);
        permutations!($Ax0, $Ax4, [$Ax1, $Ax2, $Ax3]);
    };
    ($Ax0:tt, $Ax1:tt, [$Ax2:tt, $Ax3:tt, $Ax4:tt]) => {
        permutations!($Ax0, $Ax1, $Ax2, [$Ax3, $Ax4]);
        permutations!($Ax0, $Ax1, $Ax3, [$Ax2, $Ax4]);
        permutations!($Ax0, $Ax1, $Ax4, [$Ax2, $Ax3]);
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, [$Ax3:tt, $Ax4:tt]) => {
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4);
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax4, $Ax3);
    };

    ([$Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt]) => {
        permutations!($Ax0, [$Ax1, $Ax2, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax1, [$Ax0, $Ax2, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax2, [$Ax0, $Ax1, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax3, [$Ax0, $Ax1, $Ax2, $Ax4, $Ax5]);
        permutations!($Ax4, [$Ax0, $Ax1, $Ax2, $Ax3, $Ax5]);
        permutations!($Ax5, [$Ax0, $Ax1, $Ax2, $Ax3, $Ax4]);
    };
    ($Ax0:tt, [$Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt]) => {
        permutations!($Ax0, $Ax1, [$Ax2, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax2, [$Ax1, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax3, [$Ax1, $Ax2, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax4, [$Ax1, $Ax2, $Ax3, $Ax5]);
        permutations!($Ax0, $Ax5, [$Ax1, $Ax2, $Ax3, $Ax4]);
    };
    ($Ax0:tt, $Ax1:tt, [$Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt]) => {
        permutations!($Ax0, $Ax1, $Ax2, [$Ax3, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax3, [$Ax2, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax4, [$Ax2, $Ax3, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax5, [$Ax2, $Ax3, $Ax4]);
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, [$Ax3:tt, $Ax4:tt, $Ax5:tt]) => {
        permutations!($Ax0, $Ax1, $Ax2, $Ax3, [$Ax4, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax2, $Ax4, [$Ax3, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax2, $Ax5, [$Ax3, $Ax4]);
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, [$Ax4:tt, $Ax5:tt]) => {
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5);
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax3, $Ax5, $Ax4);
    };
}

permutations!([0, 1]);
permutations!([0, 1, 2]);
permutations!([0, 1, 2, 3]);
permutations!([0, 1, 2, 3, 4]);
permutations!([0, 1, 2, 3, 4, 5]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutes() {
        let src = (Dyn::<'B'>(1), Const::<2>, Dyn::<'C'>(3), Const::<4>);

        let dst = PermuteStridesTo::<_, Axes4<1, 0, 3, 2>>::permuted(&src);
        assert_eq!(dst, (Const::<2>, Dyn::<'B'>(1), Const::<4>, Dyn::<'C'>(3)));

        let dst = PermuteStridesTo::<_, Axes4<2, 3, 0, 1>>::permuted(&src);
        assert_eq!(dst, (Dyn::<'C'>(3), Const::<4>, Dyn::<'B'>(1), Const::<2>));
    }

    #[test]
    fn test_permute_strides() {
        let src = (Dyn::<'B'>(1), Const::<2>, Dyn::<'C'>(3));
        let dst_strides =
            PermuteStridesTo::<_, Axes3<1, 2, 0>>::permute_strides(&src, src.strides());
        assert_eq!(src.strides(), [6, 3, 1]);
        assert_eq!(dst_strides, [3, 1, 6]);
    }
}
