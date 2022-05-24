//! Collection of traits to describe Nd arrays.

/// Represents something with a compile time known number of elements
pub trait CountElements {
    const NUM_ELEMENTS: usize;
    type Dtype: Clone + Default;
    const NUM_BYTES: usize = Self::NUM_ELEMENTS * std::mem::size_of::<Self::Dtype>();
}

impl CountElements for f32 {
    const NUM_ELEMENTS: usize = 1;
    type Dtype = Self;
}

impl<T: CountElements, const M: usize> CountElements for [T; M] {
    const NUM_ELEMENTS: usize = M * T::NUM_ELEMENTS;
    type Dtype = T::Dtype;
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
    type Array: 'static + Sized + Clone + CountElements<Dtype = Self::Dtype> + ZeroElements;
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
}
