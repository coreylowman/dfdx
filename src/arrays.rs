//! Collection of traits to describe Nd arrays.

/// Represents something with a compile time known number of elements
pub trait CountElements: Clone {
    type Dtype: Clone + Default;
    const NUM_ELEMENTS: usize;
    const NUM_BYTES: usize = Self::NUM_ELEMENTS * std::mem::size_of::<Self::Dtype>();

    fn ref_first_elem(&self) -> &Self::Dtype;
    fn mut_first_elem(&mut self) -> &mut Self::Dtype;
}

impl CountElements for f32 {
    type Dtype = Self;
    const NUM_ELEMENTS: usize = 1;

    fn ref_first_elem(&self) -> &Self::Dtype {
        self
    }

    fn mut_first_elem(&mut self) -> &mut Self::Dtype {
        self
    }
}

impl CountElements for usize {
    type Dtype = Self;
    const NUM_ELEMENTS: usize = 1;

    fn ref_first_elem(&self) -> &Self::Dtype {
        self
    }

    fn mut_first_elem(&mut self) -> &mut Self::Dtype {
        self
    }
}

impl<T: CountElements, const M: usize> CountElements for [T; M] {
    type Dtype = T::Dtype;
    const NUM_ELEMENTS: usize = M * T::NUM_ELEMENTS;

    fn ref_first_elem(&self) -> &Self::Dtype {
        self[0].ref_first_elem()
    }

    fn mut_first_elem(&mut self) -> &mut Self::Dtype {
        self[0].mut_first_elem()
    }
}

/// A single axis known at compile time
pub struct Axis<const I: isize>;

/// Two axes known at compile time.
pub type Axes2<const I: isize, const J: isize> = (Axis<I>, Axis<J>);

/// Three axes known at compile time.
pub type Axes3<const I: isize, const J: isize, const K: isize> = (Axis<I>, Axis<J>, Axis<K>);

/// Four axes known at compile time.
pub type Axes4<const I: isize, const J: isize, const K: isize, const L: isize> =
    (Axis<I>, Axis<J>, Axis<K>, Axis<L>);

/// Five axes known at compile time.
pub type Axes5<const I: isize, const J: isize, const K: isize, const L: isize, const M: isize> =
    (Axis<I>, Axis<J>, Axis<K>, Axis<L>, Axis<M>);

/// Six axes known at compile time.
pub type Axes6<
    const I: isize,
    const J: isize,
    const K: isize,
    const L: isize,
    const M: isize,
    const N: isize,
> = (Axis<I>, Axis<J>, Axis<K>, Axis<L>, Axis<M>, Axis<N>);

/// Represents all available axes on a tensor.
pub struct AllAxes;

/// An NdArray that has an `I`th axis
pub trait HasAxes<Axes> {
    /// The size of the axis. E.g. an nd array of shape (M, N, O):
    /// 1. The `0`th axis has `SIZE` = M
    /// 2. The `1`th axis has `SIZE` = N
    /// 3. The `2`th axis has `SIZE` = O
    const SIZE: usize;
}

macro_rules! impl_has_axis {
    ($SrcTy:tt, $Axis:expr, $Size:expr, {$($Vars:tt),*}) => {
impl<$(const $Vars: usize, )*> HasAxes<Axis<$Axis>> for $SrcTy {
    const SIZE: usize = $Size;
}
    };
}

impl_has_axis!(f32, 0, 1, {});
impl_has_axis!([f32; M], 0, M, { M });
impl_has_axis!([[f32; N]; M], 0, M, {M, N});
impl_has_axis!([[f32; N]; M], 1, N, {M, N});
impl_has_axis!([[[f32; O]; N]; M], 0, M, {M, N, O});
impl_has_axis!([[[f32; O]; N]; M], 1, N, {M, N, O});
impl_has_axis!([[[f32; O]; N]; M], 2, O, {M, N, O});
impl_has_axis!([[[[f32; P]; O]; N]; M], 0, M, {M, N, O, P});
impl_has_axis!([[[[f32; P]; O]; N]; M], 1, N, {M, N, O, P});
impl_has_axis!([[[[f32; P]; O]; N]; M], 2, O, {M, N, O, P});
impl_has_axis!([[[[f32; P]; O]; N]; M], 3, P, {M, N, O, P});
impl_has_axis!([[[[[f32; Q]; P]; O]; N]; M], 0, M, {M, N, O, P, Q});
impl_has_axis!([[[[[f32; Q]; P]; O]; N]; M], 1, N, {M, N, O, P, Q});
impl_has_axis!([[[[[f32; Q]; P]; O]; N]; M], 2, O, {M, N, O, P, Q});
impl_has_axis!([[[[[f32; Q]; P]; O]; N]; M], 3, P, {M, N, O, P, Q});
impl_has_axis!([[[[[f32; Q]; P]; O]; N]; M], 4, Q, {M, N, O, P, Q});
impl_has_axis!([[[[[[f32; R]; Q]; P]; O]; N]; M], 0, M, {M, N, O, P, Q, R});
impl_has_axis!([[[[[[f32; R]; Q]; P]; O]; N]; M], 1, N, {M, N, O, P, Q, R});
impl_has_axis!([[[[[[f32; R]; Q]; P]; O]; N]; M], 2, O, {M, N, O, P, Q, R});
impl_has_axis!([[[[[[f32; R]; Q]; P]; O]; N]; M], 3, P, {M, N, O, P, Q, R});
impl_has_axis!([[[[[[f32; R]; Q]; P]; O]; N]; M], 4, Q, {M, N, O, P, Q, R});
impl_has_axis!([[[[[[f32; R]; Q]; P]; O]; N]; M], 5, R, {M, N, O, P, Q, R});

impl<T: CountElements> HasAxes<AllAxes> for T {
    const SIZE: usize = T::NUM_ELEMENTS;
}

impl<T, const I: isize, const J: isize> HasAxes<Axes2<I, J>> for T
where
    T: HasAxes<Axis<I>> + HasAxes<Axis<J>>,
{
    const SIZE: usize = <T as HasAxes<Axis<I>>>::SIZE * <T as HasAxes<Axis<J>>>::SIZE;
}

impl<T, const I: isize, const J: isize, const K: isize> HasAxes<Axes3<I, J, K>> for T
where
    T: HasAxes<Axis<I>> + HasAxes<Axis<J>> + HasAxes<Axis<K>>,
{
    const SIZE: usize = <T as HasAxes<Axis<I>>>::SIZE
        * <T as HasAxes<Axis<J>>>::SIZE
        * <T as HasAxes<Axis<K>>>::SIZE;
}

impl<T, const I: isize, const J: isize, const K: isize, const L: isize> HasAxes<Axes4<I, J, K, L>>
    for T
where
    T: HasAxes<Axis<I>> + HasAxes<Axis<J>> + HasAxes<Axis<K>> + HasAxes<Axis<L>>,
{
    const SIZE: usize = <T as HasAxes<Axis<I>>>::SIZE
        * <T as HasAxes<Axis<J>>>::SIZE
        * <T as HasAxes<Axis<K>>>::SIZE
        * <T as HasAxes<Axis<L>>>::SIZE;
}

impl<T, const I: isize, const J: isize, const K: isize, const L: isize, const M: isize>
    HasAxes<Axes5<I, J, K, L, M>> for T
where
    T: HasAxes<Axis<I>> + HasAxes<Axis<J>> + HasAxes<Axis<K>> + HasAxes<Axis<L>> + HasAxes<Axis<M>>,
{
    const SIZE: usize = <T as HasAxes<Axis<I>>>::SIZE
        * <T as HasAxes<Axis<J>>>::SIZE
        * <T as HasAxes<Axis<K>>>::SIZE
        * <T as HasAxes<Axis<L>>>::SIZE
        * <T as HasAxes<Axis<M>>>::SIZE;
}

impl<
        T,
        const I: isize,
        const J: isize,
        const K: isize,
        const L: isize,
        const M: isize,
        const N: isize,
    > HasAxes<Axes6<I, J, K, L, M, N>> for T
where
    T: HasAxes<Axis<I>>
        + HasAxes<Axis<J>>
        + HasAxes<Axis<K>>
        + HasAxes<Axis<L>>
        + HasAxes<Axis<M>>
        + HasAxes<Axis<N>>,
{
    const SIZE: usize = <T as HasAxes<Axis<I>>>::SIZE
        * <T as HasAxes<Axis<J>>>::SIZE
        * <T as HasAxes<Axis<K>>>::SIZE
        * <T as HasAxes<Axis<L>>>::SIZE
        * <T as HasAxes<Axis<M>>>::SIZE
        * <T as HasAxes<Axis<N>>>::SIZE;
}

/// Holds an axis that represents the last (or right most) axis.
pub trait HasLastAxis {
    type LastAxis;
    const SIZE: usize;
}

impl HasLastAxis for f32 {
    type LastAxis = AllAxes;
    const SIZE: usize = 1;
}
impl<const M: usize> HasLastAxis for [f32; M] {
    type LastAxis = AllAxes;
    const SIZE: usize = M;
}
impl<const M: usize, const N: usize> HasLastAxis for [[f32; N]; M] {
    type LastAxis = Axis<1>;
    const SIZE: usize = N;
}
impl<const M: usize, const N: usize, const O: usize> HasLastAxis for [[[f32; O]; N]; M] {
    type LastAxis = Axis<2>;
    const SIZE: usize = O;
}
impl<const M: usize, const N: usize, const O: usize, const P: usize> HasLastAxis
    for [[[[f32; P]; O]; N]; M]
{
    type LastAxis = Axis<3>;
    const SIZE: usize = P;
}
impl<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize> HasLastAxis
    for [[[[[f32; Q]; P]; O]; N]; M]
{
    type LastAxis = Axis<4>;
    const SIZE: usize = Q;
}
impl<
        const M: usize,
        const N: usize,
        const O: usize,
        const P: usize,
        const Q: usize,
        const R: usize,
    > HasLastAxis for [[[[[[f32; R]; Q]; P]; O]; N]; M]
{
    type LastAxis = Axis<5>;
    const SIZE: usize = R;
}

/// Something that has compile time known zero values.
pub trait ZeroElements {
    const ZEROS: Self;
}

impl ZeroElements for f32 {
    const ZEROS: Self = 0.0;
}

impl<T: ZeroElements, const M: usize> ZeroElements for [T; M] {
    const ZEROS: Self = [T::ZEROS; M];
}

/// Has an associated type that implemented [CountElements] and [ZeroElements].
pub trait HasArrayType {
    type Dtype;
    type Array: 'static
        + Sized
        + Clone
        + CountElements<Dtype = Self::Dtype>
        + ZeroElements
        + HasAxes<Axis<0>>
        + HasAxes<AllAxes>
        + HasLastAxis;
}

/// Something that has [HasArrayType], and also can return a reference to or mutate `Self::Array`.
pub trait HasArrayData: HasArrayType {
    fn data(&self) -> &Self::Array;
    fn mut_data(&mut self) -> &mut Self::Array;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0d_count() {
        assert_eq!(1, f32::NUM_ELEMENTS);
    }

    #[test]
    fn test_1d_count() {
        assert_eq!(5, <[f32; 5]>::NUM_ELEMENTS);
    }

    #[test]
    fn test_2d_count() {
        assert_eq!(15, <[[f32; 3]; 5]>::NUM_ELEMENTS);
    }

    #[test]
    fn test_3d_count() {
        assert_eq!(30, <[[[f32; 2]; 3]; 5]>::NUM_ELEMENTS);
    }

    #[test]
    fn test_first_elem_ref() {
        let mut a: [[f32; 2]; 3] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert_eq!(a.ref_first_elem(), &1.0);
        assert_eq!(a.mut_first_elem(), &mut 1.0);
    }
}
