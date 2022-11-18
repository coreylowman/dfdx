use super::*;

pub trait PermuteShapeTo<Dst, Axes> {
    fn permuted(&self) -> Dst;
}

#[rustfmt::skip]
macro_rules! dim { (0) => { D1 }; (1) => { D2 }; (2) => { D3 }; (3) => { D4 }; (4) => { D5 }; (5) => { D6 }; }

macro_rules! impl_permute {
    ($Ax0:tt, $Ax1:tt) => {
        impl<D1: Dim, D2: Dim> PermuteShapeTo<(dim!($Ax0), dim!($Ax1)), Axes2<$Ax0, $Ax1>>
            for (D1, D2)
        {
            fn permuted(&self) -> (dim!($Ax0), dim!($Ax1)) {
                (self.$Ax0, self.$Ax1)
            }
        }
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt) => {
        impl<D1: Dim, D2: Dim, D3: Dim>
            PermuteShapeTo<(dim!($Ax0), dim!($Ax1), dim!($Ax2)), Axes3<$Ax0, $Ax1, $Ax2>>
            for (D1, D2, D3)
        {
            fn permuted(&self) -> (dim!($Ax0), dim!($Ax1), dim!($Ax2)) {
                (self.$Ax0, self.$Ax1, self.$Ax2)
            }
        }
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt) => {
        impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim>
            PermuteShapeTo<
                (dim!($Ax0), dim!($Ax1), dim!($Ax2), dim!($Ax3)),
                Axes4<$Ax0, $Ax1, $Ax2, $Ax3>,
            > for (D1, D2, D3, D4)
        {
            fn permuted(&self) -> (dim!($Ax0), dim!($Ax1), dim!($Ax2), dim!($Ax3)) {
                (self.$Ax0, self.$Ax1, self.$Ax2, self.$Ax3)
            }
        }
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt) => {
        impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim>
            PermuteShapeTo<
                (dim!($Ax0), dim!($Ax1), dim!($Ax2), dim!($Ax3), dim!($Ax4)),
                Axes5<$Ax0, $Ax1, $Ax2, $Ax3, $Ax4>,
            > for (D1, D2, D3, D4, D5)
        {
            fn permuted(&self) -> (dim!($Ax0), dim!($Ax1), dim!($Ax2), dim!($Ax3), dim!($Ax4)) {
                (self.$Ax0, self.$Ax1, self.$Ax2, self.$Ax3, self.$Ax4)
            }
        }
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt) => {
        impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim>
            PermuteShapeTo<
                (
                    dim!($Ax0),
                    dim!($Ax1),
                    dim!($Ax2),
                    dim!($Ax3),
                    dim!($Ax4),
                    dim!($Ax5),
                ),
                Axes6<$Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5>,
            > for (D1, D2, D3, D4, D5, D6)
        {
            fn permuted(
                &self,
            ) -> (
                dim!($Ax0),
                dim!($Ax1),
                dim!($Ax2),
                dim!($Ax3),
                dim!($Ax4),
                dim!($Ax5),
            ) {
                (
                    self.$Ax0, self.$Ax1, self.$Ax2, self.$Ax3, self.$Ax4, self.$Ax5,
                )
            }
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
