//! Module for data type related traits and structs. Contains things like [Unit], [Dtype], and [AMP].
//!
//! When the `f16` feature is enabled, this exports the [f16] type.
//!
//! # AMP
//!
//! [AMP](https://pytorch.org/docs/stable/amp.html) is a technique for mixed precision training.
//! This is a data type in dfdx, you can use it like any normal dtype like [`AMP<f16>`] or [`AMP<bf16>`].

mod amp;

pub use amp::AMP;

#[cfg(feature = "f16")]
pub use half::f16;

#[cfg(feature = "complex")]
pub mod complex {
    use core::ops::{Deref, DerefMut};

    #[cfg(feature = "cuda")]
    use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
    use num_complex::Complex32;
    use num_traits::{FromPrimitive, ToPrimitive};

    #[derive(PartialEq, Debug, Default, Clone, Copy)]
    pub struct Complex(Complex32);
    impl Deref for Complex {
        type Target = Complex32;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl DerefMut for Complex {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }
    const fn c1() -> Complex {
        Complex(Complex32 { re: 1.0, im: 0.0 })
    }
    impl Complex {
        pub const ONE: Complex = c1();
        pub fn new(r: f32, i: f32) -> Self {
            Self(num_complex::Complex { re: r, im: i })
        }
    }
    impl FromPrimitive for Complex {
        fn from_i64(n: i64) -> Option<Self> {
            Some(Complex(Complex32::from_i64(n)?))
        }

        fn from_u64(n: u64) -> Option<Self> {
            Some(Complex(Complex32::from_u64(n)?))
        }
    }
    impl ToPrimitive for Complex {
        fn to_i64(&self) -> Option<i64> {
            self.0.to_i64()
        }

        fn to_u64(&self) -> Option<u64> {
            self.0.to_u64()
        }
    }

    impl std::ops::Add<Self> for Complex {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }
    impl std::ops::Sub<Self> for Complex {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 - rhs.0)
        }
    }
    impl std::ops::Mul<Self> for Complex {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Self(self.0 * rhs.0)
        }
    }
    impl std::ops::Div<Self> for Complex {
        type Output = Self;

        fn div(self, rhs: Self) -> Self::Output {
            Self(self.0 / rhs.0)
        }
    }
    impl std::ops::AddAssign for Complex {
        fn add_assign(&mut self, rhs: Self) {
            self.0.add_assign(rhs.0)
        }
    }
    impl std::ops::SubAssign for Complex {
        fn sub_assign(&mut self, rhs: Self) {
            self.0.sub_assign(rhs.0)
        }
    }
    impl std::ops::MulAssign for Complex {
        fn mul_assign(&mut self, rhs: Self) {
            self.0.mul_assign(rhs.0)
        }
    }
    impl std::ops::DivAssign for Complex {
        fn div_assign(&mut self, rhs: Self) {
            self.0.div_assign(rhs.0)
        }
    }
    #[cfg(feature = "cuda")]
    unsafe impl ValidAsZeroBits for Complex {}
    #[cfg(feature = "cuda")]
    unsafe impl DeviceRepr for Complex {}
}

/// Represents a type where all 0 bits is a valid pattern.
#[cfg(not(feature = "cuda"))]
pub trait SafeZeros {}

/// Represents a type where all 0 bits is a valid pattern.
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
    // + PartialOrd
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
unit!(f16, f16::ONE);
#[cfg(feature = "complex")]
unit!(complex::Complex, complex::Complex::ONE);

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
impl Dtype for f16 {}
#[cfg(feature = "complex")]
impl Dtype for complex::Complex {}

/// Represents something that has a [Dtype].
pub trait HasDtype {
    type Dtype: Dtype;
}

/// Marker trait for types that are **not** [AMP].
pub trait NotMixedPrecision {}
impl NotMixedPrecision for f32 {}
impl NotMixedPrecision for f64 {}
impl NotMixedPrecision for i8 {}
impl NotMixedPrecision for i16 {}
impl NotMixedPrecision for i32 {}
impl NotMixedPrecision for i64 {}
impl NotMixedPrecision for i128 {}
impl NotMixedPrecision for isize {}
impl NotMixedPrecision for u8 {}
impl NotMixedPrecision for u16 {}
impl NotMixedPrecision for u32 {}
impl NotMixedPrecision for u64 {}
impl NotMixedPrecision for u128 {}
impl NotMixedPrecision for usize {}
#[cfg(feature = "f16")]
impl NotMixedPrecision for f16 {}
#[cfg(feature = "complex")]
impl NotMixedPrecision for complex::Complex {}
