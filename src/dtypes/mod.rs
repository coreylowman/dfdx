mod amp;

pub use amp::AMP;

#[cfg(feature = "f16")]
pub use half::f16;

#[cfg(not(feature = "cuda"))]
pub trait SafeZeros {}

#[cfg(feature = "cuda")]
pub trait SafeZeros: cudarc::driver::ValidAsZeroBits + cudarc::driver::DeviceRepr {}

/// Represents a unit type, but no arithmetic.
pub trait Unit:
    'static
    + Copy
    + Clone
    + Default
    + std::fmt::Debug
    + PartialEq
    + PartialOrd
    + Send
    + Sync
    + std::marker::Unpin
    + SafeZeros
{
    const ONE: Self;
}

macro_rules! unit {
    ($type:ty, $one:expr) => {
        impl SafeZeros for $type {}
        impl Unit for $type {
            const ONE: Self = $one;
        }
    };
}

unit!(f32, 1.0);
unit!(f64, 1.0);
unit!(usize, 1);
unit!(isize, 1);
unit!(u8, 1);
unit!(i8, 1);
unit!(u16, 1);
unit!(i16, 1);
unit!(u32, 1);
unit!(i32, 1);
unit!(u64, 1);
unit!(i64, 1);
unit!(u128, 1);
unit!(i128, 1);
unit!(bool, true);
#[cfg(feature = "f16")]
unit!(half::f16, half::f16::ONE);

/// Represents something that has a [Unit].
pub trait HasUnitType {
    type Unit: Unit;
}

/// Represents a data type or element of an array that can have
/// arithmatic operations applied to it. The main difference
/// between [Dtype] and [Unit] is that [`bool`] is [Unit], but
/// not [Dtype].
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
    + num_traits::FromPrimitive
    + num_traits::ToPrimitive
{
}
impl Dtype for f32 {}
impl Dtype for f64 {}
impl Dtype for i8 {}
impl Dtype for i16 {}
impl Dtype for i32 {}
impl Dtype for i64 {}
impl Dtype for i128 {}
impl Dtype for isize {}
impl Dtype for u8 {}
impl Dtype for u16 {}
impl Dtype for u32 {}
impl Dtype for u64 {}
impl Dtype for u128 {}
impl Dtype for usize {}
#[cfg(feature = "f16")]
impl Dtype for half::f16 {}

/// Represents something that has a [Dtype].
pub trait HasDtype {
    type Dtype: Dtype;
}
