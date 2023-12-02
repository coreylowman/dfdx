use crate::{shapes::*, tensor::*};

/// Concatenate two shapes along a given axis.
///
/// # [Const] dims **requires nightly**
///
/// Along Axis 0:
/// ```ignore
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a: Rank2<3, 4> = (Const, Const);
/// let b: Rank2<3, 4> = (Const, Const);
/// let _: Rank2<6, 4> = (a, b).concat_shape_along(Axis::<0>);
/// ```
///
/// Along Axis 1:
/// ```ignore
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a: Rank2<3, 4> = (Const, Const);
/// let b: Rank2<3, 4> = (Const, Const);
/// let _: Rank2<3, 8> = (a, b).concat_shape_along(Axis::<1>);
/// ```
///
/// # [usize] dims
/// Along Axis 0:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a: (usize, Const<3>) = (2, Const);
/// let b: (usize, Const<3>) = (4, Const);
/// let c: (usize, Const<3>) = (a, b).concat_shape_along(Axis::<0>);
/// assert_eq!(c, (6, Const::<3>));
/// ```
///
/// Along Axis 1:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a: (Const<2>, usize) = (Const, 2);
/// let b: (Const<2>, usize) = (Const, 4);
/// let c: (Const<2>, usize) = (a, b).concat_shape_along(Axis::<1>);
/// assert_eq!(c, (Const::<2>, 6));
/// ```
pub trait TryConcatShapeAlong<Ax>: Sized {
    type Output: Shape;

    /// Concatenates self along the given axis.
    fn concat_shape_along(self, ax: Ax) -> Self::Output {
        self.try_concat_shape_along(ax).unwrap()
    }
    /// Fallibly concatenates self along the given axis.
    fn try_concat_shape_along(self, ax: Ax) -> Result<Self::Output, Error>;
}

macro_rules! impl_concat {
    ($Ax:expr, $NumDims:expr, [$($Head:tt),*], [$($Tail:tt),*]) => {
        impl<A: Dim, B: Dim, $($Head: Dim, )* $($Tail: Dim, )*> TryConcatShapeAlong<Axis<$Ax>>
            for (
                ($($Head, )* A, $($Tail, )*),
                ($($Head, )* B, $($Tail, )*),
            )
        where
            A: std::ops::Add<B>,
            <A as std::ops::Add<B>>::Output: Dim,
            {
                type Output = (
                    $($Head, )*
                    <A as std::ops::Add<B>>::Output,
                    $($Tail, )*
                );

                fn try_concat_shape_along(self, _: Axis<$Ax>) -> Result<Self::Output, Error> {
                    let (lhs, rhs) = self;
                    let lhs_dims = lhs.concrete();
                    let rhs_dims = rhs.concrete();
                    for i in 0..$NumDims {
                        if i != $Ax {
                            assert_eq!(lhs_dims[i], rhs_dims[i]);
                        }
                    }
                    let mut out_dims = lhs_dims;
                    out_dims[$Ax] += rhs_dims[$Ax];
                    Ok(Self::Output::from_concrete(&out_dims).unwrap())
                }
            }
    };
}

impl_concat!(0, 1, [], []);
impl_concat!(0, 2, [], [D1]);
impl_concat!(0, 3, [], [D1, D2]);
impl_concat!(0, 4, [], [D1, D2, D3]);
impl_concat!(0, 5, [], [D1, D2, D3, D4]);
impl_concat!(0, 6, [], [D1, D2, D3, D4, D5]);

impl_concat!(1, 2, [D0], []);
impl_concat!(1, 3, [D0], [D2]);
impl_concat!(1, 4, [D0], [D2, D3]);
impl_concat!(1, 5, [D0], [D2, D3, D4]);
impl_concat!(1, 6, [D0], [D2, D3, D4, D5]);

impl_concat!(2, 3, [D0, D1], []);
impl_concat!(2, 4, [D0, D1], [D3]);
impl_concat!(2, 5, [D0, D1], [D3, D4]);
impl_concat!(2, 6, [D0, D1], [D3, D4, D5]);

impl_concat!(3, 4, [D0, D1, D2], []);
impl_concat!(3, 5, [D0, D1, D2], [D4]);
impl_concat!(3, 6, [D0, D1, D2], [D4, D5]);

impl_concat!(4, 5, [D0, D1, D2, D3], []);
impl_concat!(4, 6, [D0, D1, D2, D3], [D5]);

impl_concat!(5, 6, [D0, D1, D2, D3, D4], []);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat_shape() {
        let a: (usize, Const<5>) = (5, Const);
        let b: (usize, Const<5>) = (3, Const);
        assert_eq!((a, b).concat_shape_along(Axis::<0>), (8, Const::<5>));

        let a: (Const<5>, Const<5>) = (Const, Const);
        let b: (usize, Const<5>) = (3, Const);
        assert_eq!((a, b).concat_shape_along(Axis::<0>), (8, Const::<5>));

        let a: (usize, Const<5>) = (5, Const);
        let b: (Const<3>, Const<5>) = (Const, Const);
        assert_eq!((a, b).concat_shape_along(Axis::<0>), (8, Const::<5>));

        #[cfg(feature = "nightly")]
        {
            let a: (Const<5>, Const<5>) = (Const, Const);
            let b: (Const<3>, Const<5>) = (Const, Const);
            assert_eq!(
                (a, b).concat_shape_along(Axis::<0>),
                (Const::<8>, Const::<5>)
            );
        }
    }

    #[test]
    #[should_panic = "left: 10\n right: 7"]
    fn test_concat_shape_fails() {
        let a = (5, 10);
        let b = (3, 7);
        (a, b).concat_shape_along(Axis::<0>);
    }
}
