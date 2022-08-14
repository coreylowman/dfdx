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

/// TODO
pub trait HasAxis<const I: isize> {
    /// TODO
    const SIZE: usize;
}

macro_rules! impl_has_axis {
    ($SrcTy:tt, $Axis:expr, $Size:expr, {$($Vars:tt),*}) => {
impl<$(const $Vars: usize, )*> HasAxis<$Axis> for $SrcTy {
    const SIZE: usize = $Size;
}
    };
}

impl_has_axis!(f32, 0, 1, {});
impl_has_axis!(f32, -1, 1, {});
impl_has_axis!([f32; M], 0, M, { M });
impl_has_axis!([f32; M], -1, M, { M });
impl_has_axis!([[f32; N]; M], 0, M, {M, N});
impl_has_axis!([[f32; N]; M], 1, N, {M, N});
impl_has_axis!([[f32; N]; M], -1, N, {M, N});
impl_has_axis!([[[f32; O]; N]; M], 0, M, {M, N, O});
impl_has_axis!([[[f32; O]; N]; M], 1, N, {M, N, O});
impl_has_axis!([[[f32; O]; N]; M], 2, O, {M, N, O});
impl_has_axis!([[[f32; O]; N]; M], -1, O, {M, N, O});
impl_has_axis!([[[[f32; P]; O]; N]; M], 0, M, {M, N, O, P});
impl_has_axis!([[[[f32; P]; O]; N]; M], 1, N, {M, N, O, P});
impl_has_axis!([[[[f32; P]; O]; N]; M], 2, O, {M, N, O, P});
impl_has_axis!([[[[f32; P]; O]; N]; M], 3, P, {M, N, O, P});
impl_has_axis!([[[[f32; P]; O]; N]; M], -1, P, {M, N, O, P});

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
        + HasAxis<0>
        + HasAxis<-1>;
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
