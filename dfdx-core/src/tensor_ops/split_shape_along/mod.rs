use crate::{shapes::*, tensor::*};

/// Split a shape in two along a given axis.
///
/// # [Const] dims **requires nightly**
///
/// Along Axis 0:
/// ```ignore
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let (a, b): (Rank2<3, 3>, Rank2<4, 3>) = (Const::<7>, Const::<3>).split_shape_along(Axis::<0>, Const::<3>, Const::<4>);
/// ```
///
/// Along Axis 1:
/// ```ignore
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let (a, b): (Rank2<7, 2>, Rank2<7, 1>) = (Const::<7>, Const::<3>).split_shape_along(Axis::<1>, Const::<2>, Const::<1>);
/// ```
///
/// # [usize] dims
/// Along Axis 0:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let (a, b) = (7, Const::<3>).split_shape_along(Axis::<0>, 3, 4);
/// assert_eq!(a, (3, Const::<3>));
/// assert_eq!(b, (4, Const::<3>));
/// ```
///
/// Along Axis 1:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let (a, b) = (Const::<7>, 3).split_shape_along(Axis::<1>, 2, 1);
/// assert_eq!(a, (Const::<7>, 2));
/// assert_eq!(b, (Const::<7>, 1));
/// ```
pub trait TrySplitShapeAlong<Ax, A: Dim, B: Dim>: Shape {
    type Output;

    /// Splits self along the given axis.
    fn split_shape_along(self, ax: Ax, a: A, b: B) -> Self::Output {
        self.try_split_shape_along(ax, a, b).unwrap()
    }
    /// Fallibly splits self along the given axis.
    fn try_split_shape_along(self, ax: Ax, a: A, b: B) -> Result<Self::Output, Error>;
}

macro_rules! impl_split {
    ($Ax:expr, $NumDims:expr, [$($Head:tt),*], [$($Tail:tt),*]) => {
        impl<A: Dim, B: Dim, AB:Dim, $($Head: Dim, )* $($Tail: Dim, )*> TrySplitShapeAlong<Axis<$Ax>, A, B>
            for
            (
                $($Head, )*
                AB,
                $($Tail, )*
            )
        where
            ($($Head, )* A, $($Tail, )*): Shape<Concrete = <Self as Shape>::Concrete>,
            ($($Head, )* B, $($Tail, )*): Shape<Concrete = <Self as Shape>::Concrete>,
            {
                type Output =
                (
                    ($($Head, )* A, $($Tail, )*),
                    ($($Head, )* B, $($Tail, )*),
                );

                fn try_split_shape_along(self, _: Axis<$Ax>, a: A, b: B) -> Result<Self::Output, Error> {
                    let dims = self.concrete();
                    let mut lhs_dims = dims;
                    let mut rhs_dims = dims;
                    lhs_dims[$Ax] = a.size();
                    rhs_dims[$Ax] = b.size();
                    assert_eq!(dims[$Ax], lhs_dims[$Ax] + rhs_dims[$Ax]);

                    Ok((
                        <($($Head, )* A, $($Tail, )*)>::from_concrete(&lhs_dims).unwrap(),
                        <($($Head, )* B, $($Tail, )*)>::from_concrete(&rhs_dims).unwrap(),
                    ))
                }
            }
    };
}

impl_split!(0, 1, [], []);
impl_split!(0, 2, [], [D1]);
impl_split!(0, 3, [], [D1, D2]);
impl_split!(0, 4, [], [D1, D2, D3]);
impl_split!(0, 5, [], [D1, D2, D3, D4]);
impl_split!(0, 6, [], [D1, D2, D3, D4, D5]);

impl_split!(1, 2, [D0], []);
impl_split!(1, 3, [D0], [D2]);
impl_split!(1, 4, [D0], [D2, D3]);
impl_split!(1, 5, [D0], [D2, D3, D4]);
impl_split!(1, 6, [D0], [D2, D3, D4, D5]);

impl_split!(2, 3, [D0, D1], []);
impl_split!(2, 4, [D0, D1], [D3]);
impl_split!(2, 5, [D0, D1], [D3, D4]);
impl_split!(2, 6, [D0, D1], [D3, D4, D5]);

impl_split!(3, 4, [D0, D1, D2], []);
impl_split!(3, 5, [D0, D1, D2], [D4]);
impl_split!(3, 6, [D0, D1, D2], [D4, D5]);

impl_split!(4, 5, [D0, D1, D2, D3], []);
impl_split!(4, 6, [D0, D1, D2, D3], [D5]);

impl_split!(5, 6, [D0, D1, D2, D3, D4], []);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_shape() {
        let a: (usize, Const<5>) = (5, Const);
        let b: (usize, Const<5>) = (3, Const);
        assert_eq!(
            (8, Const::<5>).split_shape_along(Axis::<0>, a.0, b.0),
            (a, b)
        );

        let a: (Const<5>, Const<5>) = (Const, Const);
        let b: (usize, Const<5>) = (3, Const);
        assert_eq!(
            (8, Const::<5>).split_shape_along(Axis::<0>, a.0, b.0),
            (a, b)
        );

        let a: (usize, Const<5>) = (5, Const);
        let b: (Const<3>, Const<5>) = (Const, Const);
        assert_eq!(
            (8, Const::<5>).split_shape_along(Axis::<0>, a.0, b.0),
            (a, b)
        );

        #[cfg(feature = "nightly")]
        {
            let a: (Const<5>, Const<5>) = (Const, Const);
            let b: (Const<3>, Const<5>) = (Const, Const);
            assert_eq!(
                (Const::<8>, Const::<5>).split_shape_along(Axis::<0>, a.0, b.0),
                (a, b)
            );
        }
    }

    #[test]
    #[should_panic = "left: 8\n right: 7"]
    fn test_split_shape_fails() {
        let a: (usize, Const<5>) = (4, Const);
        let b: (usize, Const<5>) = (3, Const);
        (8, Const::<5>).split_shape_along(Axis::<0>, a.0, b.0);
    }
}
