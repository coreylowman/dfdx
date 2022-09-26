pub(crate) struct Axis<const I: isize>;
pub(crate) type Axes2<const I: isize, const J: isize> = (Axis<I>, Axis<J>);
pub(crate) type Axes3<const I: isize, const J: isize, const K: isize> = (Axis<I>, Axis<J>, Axis<K>);
pub(crate) type Axes4<const I: isize, const J: isize, const K: isize, const L: isize> =
    (Axis<I>, Axis<J>, Axis<K>, Axis<L>);
