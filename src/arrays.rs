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

/// An NdArray that is more than 0 dimensions (i.e. >= 1 dimension). This exposes the type
/// of the last dimension (inner most) of the array as a type through [MultiDimensional::LastDim].
///
/// Example:
///
/// ```rust
/// # use dfdx::prelude::*;
/// let _: [f32; 5] = <[[f32; 5]; 3] as MultiDimensional>::LastDim::default();
/// ```
pub trait MultiDimensional: CountElements {
    /// The inner most dimension of a type.
    type LastDim: CountElements<Dtype = Self::Dtype>
        + std::ops::Index<usize, Output = Self::Dtype>
        + std::ops::IndexMut<usize, Output = Self::Dtype>;
}

impl<const M: usize> MultiDimensional for [f32; M] {
    type LastDim = Self;
}

impl<T: MultiDimensional, const M: usize> MultiDimensional for [T; M] {
    type LastDim = T::LastDim;
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

    #[test]
    fn test_first_elem_ref() {
        let mut a: [[f32; 2]; 3] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert_eq!(a.ref_first_elem(), &1.0);
        assert_eq!(a.mut_first_elem(), &mut 1.0);
    }
}
