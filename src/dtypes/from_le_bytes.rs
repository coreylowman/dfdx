/// Convert slice of little endian bytes into a type.
pub trait FromLeBytes {
    fn from_le_bytes(bytes: &[u8]) -> Self;
}

impl<T: FromLeBytes> FromLeBytes for super::AMP<T> {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        super::AMP(T::from_le_bytes(bytes))
    }
}

macro_rules! from_le_bytes {
    ($type:ty) => {
        impl FromLeBytes for $type {
            fn from_le_bytes(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().unwrap())
            }
        }
    };
}

from_le_bytes!(f32);
from_le_bytes!(f64);
from_le_bytes!(u8);
from_le_bytes!(u16);
from_le_bytes!(u32);
from_le_bytes!(u64);
from_le_bytes!(u128);
from_le_bytes!(i8);
from_le_bytes!(i16);
from_le_bytes!(i32);
from_le_bytes!(i64);
from_le_bytes!(i128);
#[cfg(feature = "f16")]
from_le_bytes!(super::f16);

impl FromLeBytes for bool {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
}

impl FromLeBytes for usize {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        u64::from_le_bytes(bytes.try_into().unwrap()) as Self
    }
}

impl FromLeBytes for isize {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        i64::from_le_bytes(bytes.try_into().unwrap()) as Self
    }
}
