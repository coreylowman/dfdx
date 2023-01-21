use super::{axes::*, ReduceShapeTo};

/// Represents a unit type, but no arithmetic.
pub trait Unit:
    'static + Copy + Clone + Default + std::fmt::Debug + PartialOrd + Send + Sync + std::marker::Unpin
{
}
impl Unit for f32 {}
impl Unit for f64 {}
impl Unit for usize {}
impl Unit for u8 {}
impl Unit for u16 {}
impl Unit for u32 {}
impl Unit for u64 {}
impl Unit for u128 {}
impl Unit for isize {}
impl Unit for i8 {}
impl Unit for i16 {}
impl Unit for i32 {}
impl Unit for i64 {}
impl Unit for i128 {}
impl Unit for bool {}

/// Represents something that has a [Unit].
pub trait HasUnitType {
    type Unit: Unit;
}

/// Represents a data type or element of an array.
pub trait Dtype:
    Unit
    + std::ops::Add<Self, Output = Self>
    + std::ops::Sub<Self, Output = Self>
    + std::ops::Mul<Self, Output = Self>
    + std::ops::Div<Self, Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
{
}
impl Dtype for f32 {}
impl Dtype for f64 {}
impl Dtype for usize {}

/// Represents something that has a [Dtype].
pub trait HasDtype {
    type Dtype: Dtype;
}

/// Represents a single dimension of a multi dimensional [Shape]
pub trait Dim: 'static + Copy + Clone + std::fmt::Debug + Send + Sync + Eq + PartialEq {
    fn size(&self) -> usize;
    fn from_size(size: usize) -> Option<Self>;
}

/// Represents a single dimension where all
/// instances are guarunteed to be the same size at compile time.
pub trait ConstDim: Default + Dim {}

impl Dim for usize {
    #[inline(always)]
    fn size(&self) -> usize {
        *self
    }
    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        Some(size)
    }
}

/// Represents a [Dim] with size known at compile time
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Const<const M: usize>;
impl<const M: usize> Dim for Const<M> {
    #[inline(always)]
    fn size(&self) -> usize {
        M
    }
    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        if size == M {
            Some(Const)
        } else {
            None
        }
    }
}

impl<const M: usize> ConstDim for Const<M> {}

/// A collection of dimensions ([Dim]) that change how a multi-dimensional
/// array is interacted with.
pub trait Shape:
    'static
    + std::fmt::Debug
    + Clone
    + Copy
    + Send
    + Sync
    + Eq
    + PartialEq
    + HasAxes<Self::AllAxes>
    + HasAxes<Self::LastAxis>
    + ReduceShapeTo<(), Self::AllAxes>
{
    /// The number of dimensions the shape has
    const NUM_DIMS: usize;

    /// Is `[usize; Self::NUM_DIMS]`, but that is not usable yet.
    type Concrete: std::fmt::Debug
        + Clone
        + Copy
        + Default
        + Eq
        + PartialEq
        + std::ops::Index<usize, Output = usize>
        + std::ops::IndexMut<usize>
        + Send
        + Sync
        + IntoIterator<Item = usize>
        + Into<std::vec::Vec<usize>>;

    /// All the axes of this shape
    type AllAxes: Axes;

    /// The last axis of this shape
    type LastAxis: Axes;

    fn concrete(&self) -> Self::Concrete;
    fn from_concrete(concrete: &Self::Concrete) -> Option<Self>;

    /// The number of elements in this shape; the product of all dimensions.
    #[inline(always)]
    fn num_elements(&self) -> usize {
        self.concrete().into_iter().product()
    }

    /// The strides of how this shape is layed out in memory.
    #[inline(always)]
    fn strides(&self) -> Self::Concrete {
        let sizes = self.concrete();
        let mut strides: Self::Concrete = Default::default();
        strides[Self::NUM_DIMS - 1] = 1;
        for i in (0..(Self::NUM_DIMS - 1)).rev() {
            strides[i] = strides[i + 1] * sizes[i + 1];
        }
        strides
    }
}

/// Represents a [Shape] that has all [ConstDim]s
pub trait ConstShape: Default + Shape {}

/// Represents something that has a [Shape].
pub trait HasShape {
    type WithShape<New: Shape>: HasShape<Shape = New>;
    type Shape: Shape;
    fn shape(&self) -> &Self::Shape;
}

impl<S: Shape> HasShape for S {
    type WithShape<New: Shape> = New;
    type Shape = Self;
    fn shape(&self) -> &Self::Shape {
        self
    }
}

/// Compile time known shape with 0 dimensions
pub type Rank0 = ();
/// Compile time known shape with 1 dimensions
pub type Rank1<const M: usize> = (Const<M>,);
/// Compile time known shape with 2 dimensions
pub type Rank2<const M: usize, const N: usize> = (Const<M>, Const<N>);
/// Compile time known shape with 3 dimensions
pub type Rank3<const M: usize, const N: usize, const O: usize> = (Const<M>, Const<N>, Const<O>);
/// Compile time known shape with 4 dimensions
pub type Rank4<const M: usize, const N: usize, const O: usize, const P: usize> =
    (Const<M>, Const<N>, Const<O>, Const<P>);
/// Compile time known shape with 5 dimensions
pub type Rank5<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize> =
    (Const<M>, Const<N>, Const<O>, Const<P>, Const<Q>);
#[rustfmt::skip]
/// Compile time known shape with 6 dimensions
pub type Rank6<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize, const R: usize> =
    (Const<M>, Const<N>, Const<O>, Const<P>, Const<Q>, Const<R>);

macro_rules! shape {
    (($($D:tt $Idx:tt),*), rank=$Num:expr, all=$All:tt) => {
impl<$($D: Dim, )*> Shape for ($($D, )*) {
    const NUM_DIMS: usize = $Num;
    type Concrete = [usize; $Num];
    type AllAxes = $All<$($Idx,)*>;
    type LastAxis = Axis<{$Num - 1}>;
    #[inline(always)]
    fn concrete(&self) -> Self::Concrete {
        [$(self.$Idx.size(), )*]
    }
    #[inline(always)]
    fn from_concrete(concrete: &Self::Concrete) -> Option<Self> {
        Some(($(Dim::from_size(concrete[$Idx])?, )*))
    }
}
impl<$($D: ConstDim, )*> ConstShape for ($($D, )*) { }
    };
}

impl Shape for () {
    const NUM_DIMS: usize = 0;
    type Concrete = [usize; 0];
    type AllAxes = Axis<0>;
    type LastAxis = Axis<0>;
    #[inline(always)]
    fn concrete(&self) -> Self::Concrete {
        []
    }
    #[inline(always)]
    fn strides(&self) -> Self::Concrete {
        []
    }
    #[inline(always)]
    fn from_concrete(_: &Self::Concrete) -> Option<Self> {
        Some(())
    }
}
impl ConstShape for () {}

shape!((D1 0), rank=1, all=Axis);
shape!((D1 0, D2 1), rank=2, all=Axes2);
shape!((D1 0, D2 1, D3 2), rank=3, all=Axes3);
shape!((D1 0, D2 1, D3 2, D4 3), rank=4, all=Axes4);
shape!((D1 0, D2 1, D3 2, D4 3, D5 4), rank=5, all=Axes5);
shape!((D1 0, D2 1, D3 2, D4 3, D5 4, D6 5), rank=6, all=Axes6);
