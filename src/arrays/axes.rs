use super::shape::*;

pub trait AxesAsArray {
    type Array: IntoIterator<Item = isize>;
    fn as_array() -> Self::Array;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Axis<const I: isize>;
impl<const I: isize> AxesAsArray for Axis<I> {
    type Array = [isize; 1];
    fn as_array() -> Self::Array {
        [I]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Axes2<const I: isize, const J: isize>;
impl<const I: isize, const J: isize> AxesAsArray for Axes2<I, J> {
    type Array = [isize; 2];
    fn as_array() -> Self::Array {
        [I, J]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Axes3<const I: isize, const J: isize, const K: isize>;
impl<const I: isize, const J: isize, const K: isize> AxesAsArray for Axes3<I, J, K> {
    type Array = [isize; 3];
    fn as_array() -> Self::Array {
        [I, J, K]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Axes4<const I: isize, const J: isize, const K: isize, const L: isize>;
impl<const I: isize, const J: isize, const K: isize, const L: isize> AxesAsArray
    for Axes4<I, J, K, L>
{
    type Array = [isize; 4];
    fn as_array() -> Self::Array {
        [I, J, K, L]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Axes5<const I: isize, const J: isize, const K: isize, const L: isize, const M: isize>;
impl<const I: isize, const J: isize, const K: isize, const L: isize, const M: isize> AxesAsArray
    for Axes5<I, J, K, L, M>
{
    type Array = [isize; 5];
    fn as_array() -> Self::Array {
        [I, J, K, L, M]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Axes6<
    const I: isize,
    const J: isize,
    const K: isize,
    const L: isize,
    const M: isize,
    const N: isize,
>;
impl<
        const I: isize,
        const J: isize,
        const K: isize,
        const L: isize,
        const M: isize,
        const N: isize,
    > AxesAsArray for Axes6<I, J, K, L, M, N>
{
    type Array = [isize; 6];
    fn as_array() -> Self::Array {
        [I, J, K, L, M, N]
    }
}

pub trait HasLastAxis {
    type LastAxis;
}
macro_rules! impl_has_last_axis {
    (($($Vars:tt),*), $Axis:ty) => {
impl<$($Vars: Dim, )*> HasLastAxis for ($($Vars, )*) {
    type LastAxis = $Axis;
}
    };
}
impl_has_last_axis!((D1), Axis<0>);
impl_has_last_axis!((D1, D2), Axis<1>);
impl_has_last_axis!((D1, D2, D3), Axis<2>);
impl_has_last_axis!((D1, D2, D3, D4), Axis<3>);
impl_has_last_axis!((D1, D2, D3, D4, D5), Axis<4>);
impl_has_last_axis!((D1, D2, D3, D4, D5, D6), Axis<5>);

pub trait HasAxes<Axes> {
    fn size(&self) -> usize;
}

macro_rules! impl_has_axis {
    (($($Vars:tt),*), $Axis:tt) => {
impl<$($Vars: Dim, )*> HasAxes<Axis<$Axis>> for ($($Vars, )*) {
    fn size(&self) -> usize {
        self.$Axis.size()
    }
}
    };
}

impl HasAxes<Axis<0>> for () {
    fn size(&self) -> usize {
        1
    }
}

impl_has_axis!((D1), 0);
impl_has_axis!((D1, D2), 0);
impl_has_axis!((D1, D2), 1);
impl_has_axis!((D1, D2, D3), 0);
impl_has_axis!((D1, D2, D3), 1);
impl_has_axis!((D1, D2, D3), 2);
impl_has_axis!((D1, D2, D3, D4), 0);
impl_has_axis!((D1, D2, D3, D4), 1);
impl_has_axis!((D1, D2, D3, D4), 2);
impl_has_axis!((D1, D2, D3, D4), 3);
impl_has_axis!((D1, D2, D3, D4, D5), 0);
impl_has_axis!((D1, D2, D3, D4, D5), 1);
impl_has_axis!((D1, D2, D3, D4, D5), 2);
impl_has_axis!((D1, D2, D3, D4, D5), 3);
impl_has_axis!((D1, D2, D3, D4, D5), 4);
impl_has_axis!((D1, D2, D3, D4, D5, D6), 0);
impl_has_axis!((D1, D2, D3, D4, D5, D6), 1);
impl_has_axis!((D1, D2, D3, D4, D5, D6), 2);
impl_has_axis!((D1, D2, D3, D4, D5, D6), 3);
impl_has_axis!((D1, D2, D3, D4, D5, D6), 4);
impl_has_axis!((D1, D2, D3, D4, D5, D6), 5);

impl<const I: isize, const J: isize, S> HasAxes<Axes2<I, J>> for S
where
    Self: HasAxes<Axis<I>> + HasAxes<Axis<J>>,
{
    fn size(&self) -> usize {
        <Self as HasAxes<Axis<I>>>::size(self) * <Self as HasAxes<Axis<J>>>::size(self)
    }
}

impl<const I: isize, const J: isize, const K: isize, S> HasAxes<Axes3<I, J, K>> for S
where
    Self: HasAxes<Axis<I>> + HasAxes<Axis<J>> + HasAxes<Axis<K>>,
{
    fn size(&self) -> usize {
        <Self as HasAxes<Axis<I>>>::size(self)
            * <Self as HasAxes<Axis<J>>>::size(self)
            * <Self as HasAxes<Axis<K>>>::size(self)
    }
}

impl<const I: isize, const J: isize, const K: isize, const L: isize, S> HasAxes<Axes4<I, J, K, L>>
    for S
where
    Self: HasAxes<Axis<I>> + HasAxes<Axis<J>> + HasAxes<Axis<K>> + HasAxes<Axis<L>>,
{
    fn size(&self) -> usize {
        <Self as HasAxes<Axis<I>>>::size(self)
            * <Self as HasAxes<Axis<J>>>::size(self)
            * <Self as HasAxes<Axis<K>>>::size(self)
            * <Self as HasAxes<Axis<L>>>::size(self)
    }
}

impl<const I: isize, const J: isize, const K: isize, const L: isize, const M: isize, S>
    HasAxes<Axes5<I, J, K, L, M>> for S
where
    Self: HasAxes<Axis<I>>
        + HasAxes<Axis<J>>
        + HasAxes<Axis<K>>
        + HasAxes<Axis<L>>
        + HasAxes<Axis<M>>,
{
    fn size(&self) -> usize {
        <Self as HasAxes<Axis<I>>>::size(self)
            * <Self as HasAxes<Axis<J>>>::size(self)
            * <Self as HasAxes<Axis<K>>>::size(self)
            * <Self as HasAxes<Axis<L>>>::size(self)
            * <Self as HasAxes<Axis<M>>>::size(self)
    }
}

impl<
        const I: isize,
        const J: isize,
        const K: isize,
        const L: isize,
        const M: isize,
        const N: isize,
        S,
    > HasAxes<Axes6<I, J, K, L, M, N>> for S
where
    Self: HasAxes<Axis<I>>
        + HasAxes<Axis<J>>
        + HasAxes<Axis<K>>
        + HasAxes<Axis<L>>
        + HasAxes<Axis<M>>
        + HasAxes<Axis<N>>,
{
    fn size(&self) -> usize {
        <Self as HasAxes<Axis<I>>>::size(self)
            * <Self as HasAxes<Axis<J>>>::size(self)
            * <Self as HasAxes<Axis<K>>>::size(self)
            * <Self as HasAxes<Axis<L>>>::size(self)
            * <Self as HasAxes<Axis<M>>>::size(self)
            * <Self as HasAxes<Axis<N>>>::size(self)
    }
}
