/// Convert a type into a slice of little endian bytes.
pub trait ToLeBytes {
    type Array: IntoIterator<Item = u8>;
    fn to_le_bytes(self) -> Self::Array;
}

impl<T: ToLeBytes> ToLeBytes for super::AMP<T> {
    type Array = T::Array;
    fn to_le_bytes(self) -> Self::Array {
        self.0.to_le_bytes()
    }
}

macro_rules! to_le_bytes {
    ($Ty:ty, $Array:ty) => {
        impl ToLeBytes for $Ty {
            type Array = $Array;
            fn to_le_bytes(self) -> Self::Array {
                self.to_le_bytes()
            }
        }
    };
}

to_le_bytes!(f32, [u8; 4]);
to_le_bytes!(f64, [u8; 8]);
to_le_bytes!(u8, [u8; 1]);
to_le_bytes!(u16, [u8; 2]);
to_le_bytes!(u32, [u8; 4]);
to_le_bytes!(u64, [u8; 8]);
to_le_bytes!(u128, [u8; 16]);
to_le_bytes!(i8, [u8; 1]);
to_le_bytes!(i16, [u8; 2]);
to_le_bytes!(i32, [u8; 4]);
to_le_bytes!(i64, [u8; 8]);
to_le_bytes!(i128, [u8; 16]);
#[cfg(feature = "f16")]
to_le_bytes!(super::f16, [u8; 2]);

impl ToLeBytes for bool {
    type Array = [u8; 1];
    fn to_le_bytes(self) -> Self::Array {
        [self as u8]
    }
}

impl ToLeBytes for usize {
    type Array = [u8; 8];
    fn to_le_bytes(self) -> Self::Array {
        (self as u64).to_le_bytes()
    }
}

impl ToLeBytes for isize {
    type Array = [u8; 8];
    fn to_le_bytes(self) -> Self::Array {
        (self as i64).to_le_bytes()
    }
}
