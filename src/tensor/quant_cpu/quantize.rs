use core::cmp::Ordering;

use crate::shapes::Unit;

/// Error while creating a u4, because the value was out of range and could not be converted.
#[derive(Debug, Clone, Copy)]
pub struct HalfByteConstructionError;

impl std::fmt::Display for HalfByteConstructionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "failed to construct a half byte value")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HalfByteConstructionError {}

/// A Half-byte value. This type exists only for validation purposes, to ensure that
/// values that get stores are within the range [0..16)
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Clone, Default)]
pub struct u4(pub(crate) u8);

impl u4 {
    fn from_byte(byte_value: u8) -> Result<Self, HalfByteConstructionError> {
        if byte_value > 0b_0000_1111 {
            Err(HalfByteConstructionError)
        } else {
            Ok(Self(byte_value))
        }
    }
}

/// A pair of half-byte values stored as one byte.
/// Utilizes a trait to make usage simpler.
pub(crate) trait HalfBytePair {
    fn half_byte_pair(first: u4, second: u4) -> Self;

    /// Gets the first half-byte.
    fn first(&self) -> u4;

    /// Gets the second half-byte.
    fn second(&self) -> u4;

    /// Sets the first half-byte, leaving the rest alone.
    fn set_first(&mut self, value: u4);

    /// Sets the second half-byte, leaving the rest alone.
    fn set_second(&mut self, value: u4);
}
impl HalfBytePair for u8 {
    fn half_byte_pair(first: u4, second: u4) -> Self {
        (first.0 << 4) | second.0
    }

    fn first(&self) -> u4 {
        u4(self >> 4)
    }

    fn second(&self) -> u4 {
        u4(self & 0b_0000_1111)
    }

    fn set_first(&mut self, value: u4) {
        *self &= 0b_0000_1111;
        *self |= value.0 << 4
    }

    fn set_second(&mut self, value: u4) {
        *self &= 0b_1111_0000;
        *self |= value.0
    }
}

/// Implementation of the Q4_0 quantization method outlined [here](https://github.com/ggerganov/ggml/pull/27#).
#[derive(Copy, Clone, Debug)]
pub struct ScaledQuant<E> {
    scaling_factor: E,
}

/// Implementation of the Q4_1 quantization method outlined [here](https://github.com/ggerganov/ggml/pull/27#).
#[derive(Copy, Clone, Debug)]
pub struct OffsetQuant<E> {
    scaling_factor: E,
    offset_factor: E,
}

/// Trait that defines how to quantize a value.
pub trait HalfByteQuantizer: Clone {
    type Value: num_traits::Float + Unit;

    fn from_values(values: &[Self::Value]) -> Self
    where
        Self: Sized;

    fn quantize(&self, value: Self::Value) -> u4;

    fn dequantize(&self, half_byte: u4) -> Self::Value;
}

impl<E: num_traits::Float + Unit> HalfByteQuantizer for ScaledQuant<E> {
    type Value = E;

    fn from_values(values: &[E]) -> Self {
        Self {
            scaling_factor: values
                .iter()
                .max_by(|f1, f2| f1.partial_cmp(f2).unwrap())
                .unwrap()
                .abs()
                / E::from(7.0).unwrap(),
        }
    }

    fn quantize(&self, value: E) -> u4 {
        let inv_scaling_factor = if self.scaling_factor.is_zero() {
            E::zero()
        } else {
            E::one() / self.scaling_factor
        };
        u4::from_byte(
            ((value * inv_scaling_factor).round().to_f32().unwrap() as i8 + 8)
                .min(15)
                .max(0) as u8,
        )
        .unwrap()
    }

    fn dequantize(&self, half_byte: u4) -> E {
        E::from(half_byte.0 as i8 - 8).unwrap() * self.scaling_factor
    }
}

impl<E: num_traits::Float + Unit> HalfByteQuantizer for OffsetQuant<E> {
    type Value = E;

    fn from_values(values: &[E]) -> Self {
        let (mut min, mut max) = (E::max_value(), E::min_value());
        for v in values.iter() {
            if matches!(min.partial_cmp(v), Some(Ordering::Greater)) {
                min = *v;
            }
            if matches!(max.partial_cmp(v), Some(Ordering::Less)) {
                max = *v;
            }
        }
        Self {
            scaling_factor: (max - min) / E::from(15.0).unwrap(),
            offset_factor: min,
        }
    }

    fn quantize(&self, value: E) -> u4 {
        let inv_scaling_factor = if self.scaling_factor.is_zero() {
            E::zero()
        } else {
            E::one() / self.scaling_factor
        };
        u4::from_byte(
            (((value - self.offset_factor) * inv_scaling_factor)
                .round()
                .to_f32()
                .unwrap() as u8)
                .min(15)
                .max(0),
        )
        .unwrap()
    }

    fn dequantize(&self, half_byte: u4) -> E {
        E::from(half_byte.0).unwrap() * self.scaling_factor + self.offset_factor
    }
}
