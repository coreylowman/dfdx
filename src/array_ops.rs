pub trait ZeroElements {
    const ZEROS: Self;
}

impl ZeroElements for f32 {
    const ZEROS: Self = 0.0;
}

impl<T: ZeroElements, const M: usize> ZeroElements for [T; M] {
    const ZEROS: Self = [T::ZEROS; M];
}

pub trait CountElements {
    const NUM_ELEMENTS: usize;
}

impl CountElements for f32 {
    const NUM_ELEMENTS: usize = 1;
}

impl<T: CountElements, const M: usize> CountElements for [T; M] {
    const NUM_ELEMENTS: usize = M * T::NUM_ELEMENTS;
}

pub trait MapElements: Copy {
    fn map_elems<F: FnMut(&f32) -> f32 + Copy>(&self, f: F) -> Self;
    fn mapv_elems<F: FnMut(f32) -> f32 + Copy>(&self, f: F) -> Self;
    fn map_assign_elems<F: FnMut(&mut f32) + Copy>(&mut self, f: F);
}

impl MapElements for f32 {
    fn map_elems<F: FnMut(&f32) -> f32 + Copy>(&self, mut f: F) -> Self {
        f(self)
    }

    fn mapv_elems<F: FnMut(f32) -> f32 + Copy>(&self, mut f: F) -> Self {
        f(*self)
    }

    fn map_assign_elems<F: FnMut(&mut f32) + Copy>(&mut self, mut f: F) {
        f(self)
    }
}

impl<T: MapElements + ZeroElements, const M: usize> MapElements for [T; M] {
    fn map_elems<F: FnMut(&f32) -> f32 + Copy>(&self, f: F) -> Self {
        let mut result = Self::ZEROS;
        for i in 0..M {
            result[i] = self[i].map_elems(f);
        }
        result
    }

    fn mapv_elems<F: FnMut(f32) -> f32 + Copy>(&self, f: F) -> Self {
        let mut result = Self::ZEROS;
        for i in 0..M {
            result[i] = self[i].mapv_elems(f);
        }
        result
    }

    fn map_assign_elems<F: FnMut(&mut f32) + Copy>(&mut self, f: F) {
        for i in 0..M {
            self[i].map_assign_elems(f);
        }
    }
}

pub trait FillElements: Sized + ZeroElements {
    fn filled_with<F: FnMut() -> f32>(f: &mut F) -> Self {
        let mut result = Self::ZEROS;
        result.fill_with(f);
        result
    }
    fn fill_with<F: FnMut() -> f32>(&mut self, f: &mut F);
}

impl FillElements for f32 {
    fn fill_with<F: FnMut() -> f32>(&mut self, f: &mut F) {
        *self = f();
    }
}

impl<T: FillElements + ZeroElements, const M: usize> FillElements for [T; M] {
    fn fill_with<F: FnMut() -> f32>(&mut self, f: &mut F) {
        for i in 0..M {
            self[i].fill_with(f);
        }
    }
}

pub trait ZipMapElements {
    fn zip_map<F: FnMut(&f32, &f32) -> f32 + Copy>(&self, other: &Self, f: F) -> Self;
}

impl ZipMapElements for f32 {
    fn zip_map<F: FnMut(&f32, &f32) -> f32 + Copy>(&self, other: &Self, mut f: F) -> Self {
        f(self, other)
    }
}

impl<T: ZipMapElements + ZeroElements, const M: usize> ZipMapElements for [T; M] {
    fn zip_map<F: FnMut(&f32, &f32) -> f32 + Copy>(&self, other: &Self, f: F) -> Self {
        let mut result = Self::ZEROS;
        for i in 0..M {
            result[i] = self[i].zip_map(&other[i], f);
        }
        result
    }
}

pub trait AddElements: Sized + Copy {
    fn add(&self, rhs: &Self) -> Self {
        let mut result = *self;
        result.add_assign(rhs);
        result
    }
    fn add_assign(&mut self, rhs: &Self);
}

impl AddElements for f32 {
    fn add_assign(&mut self, rhs: &Self) {
        *self += rhs;
    }
}

impl<T: AddElements, const M: usize> AddElements for [T; M] {
    fn add_assign(&mut self, rhs: &Self) {
        for i in 0..M {
            self[i].add_assign(&rhs[i]);
        }
    }
}

pub trait SubElements<Rhs>: Sized + Copy {
    fn sub(&self, rhs: &Rhs) -> Self {
        let mut result = *self;
        result.sub_assign(rhs);
        result
    }
    fn sub_assign(&mut self, rhs: &Rhs);
}

impl SubElements<f32> for f32 {
    fn sub_assign(&mut self, rhs: &f32) {
        *self -= rhs;
    }
}

impl<const M: usize> SubElements<f32> for [f32; M] {
    fn sub_assign(&mut self, rhs: &f32) {
        for i in 0..M {
            self[i] -= rhs;
        }
    }
}

impl<Rhs, T: SubElements<Rhs>, const M: usize> SubElements<[Rhs; M]> for [T; M] {
    fn sub_assign(&mut self, rhs: &[Rhs; M]) {
        for i in 0..M {
            self[i].sub_assign(&rhs[i]);
        }
    }
}

pub trait MulElements<Rhs>: Sized + Copy {
    fn mul(&self, rhs: &Rhs) -> Self {
        let mut result = *self;
        result.mul_assign(rhs);
        result
    }
    fn mul_assign(&mut self, rhs: &Rhs);
}

impl MulElements<f32> for f32 {
    fn mul_assign(&mut self, rhs: &Self) {
        *self *= rhs;
    }
}

impl<const M: usize> MulElements<f32> for [f32; M] {
    fn mul_assign(&mut self, rhs: &f32) {
        for i in 0..M {
            self[i] *= rhs;
        }
    }
}

impl<Rhs, T: MulElements<Rhs>, const M: usize> MulElements<[Rhs; M]> for [T; M] {
    fn mul_assign(&mut self, rhs: &[Rhs; M]) {
        for i in 0..M {
            self[i].mul_assign(&rhs[i]);
        }
    }
}

pub trait ScaleElements: Sized + Copy {
    fn scale(&self, rhs: &f32) -> Self {
        let mut result = *self;
        result.scale_assign(rhs);
        result
    }
    fn scale_assign(&mut self, rhs: &f32);
}

impl ScaleElements for f32 {
    fn scale_assign(&mut self, rhs: &f32) {
        *self *= rhs;
    }
}

impl<T: ScaleElements, const M: usize> ScaleElements for [T; M] {
    fn scale_assign(&mut self, rhs: &f32) {
        for i in 0..M {
            self[i].scale_assign(rhs);
        }
    }
}
pub trait ReduceElements: CountElements {
    fn reduce<F: FnMut(f32, f32) -> f32 + Copy>(&self, f: F) -> f32;

    fn sum(&self) -> f32 {
        self.reduce(|a, b| a + b)
    }

    fn max(&self) -> f32 {
        self.reduce(f32::max)
    }

    fn min(&self) -> f32 {
        self.reduce(f32::min)
    }

    fn mean(&self) -> f32 {
        self.sum() / Self::NUM_ELEMENTS as f32
    }
}

impl ReduceElements for f32 {
    fn reduce<F: FnMut(f32, f32) -> f32 + Copy>(&self, _: F) -> f32 {
        *self
    }
}

impl<T: ReduceElements, const M: usize> ReduceElements for [T; M] {
    fn reduce<F: FnMut(f32, f32) -> f32 + Copy>(&self, f: F) -> f32 {
        (0..M).map(|i| self[i].reduce(f)).reduce(f).unwrap()
    }
}

pub trait ReduceInnerElements: Sized {
    type Output: Sized + ZeroElements;
    fn reduce_inner<F: FnMut(f32, f32) -> f32 + Copy>(&self, f: F) -> Self::Output;
}

impl<const M: usize> ReduceInnerElements for [f32; M] {
    type Output = f32;
    fn reduce_inner<F: FnMut(f32, f32) -> f32 + Copy>(&self, f: F) -> Self::Output {
        self.iter().cloned().reduce(f).unwrap()
    }
}

impl<T: ReduceInnerElements, const M: usize> ReduceInnerElements for [T; M] {
    type Output = [<T as ReduceInnerElements>::Output; M];
    fn reduce_inner<F: FnMut(f32, f32) -> f32 + Copy>(&self, f: F) -> Self::Output {
        let mut result = Self::Output::ZEROS;
        for i in 0..M {
            result[i] = self[i].reduce_inner(f);
        }
        result
    }
}

pub trait MapInnerElements {
    type Inner;
    fn map_assign_inner<F: FnMut(&mut Self::Inner) + Copy>(&mut self, f: F);
}

impl<const M: usize> MapInnerElements for [f32; M] {
    type Inner = Self;
    fn map_assign_inner<F: FnMut(&mut Self::Inner) + Copy>(&mut self, mut f: F) {
        f(self)
    }
}

impl<T: MapInnerElements, const M: usize> MapInnerElements for [T; M] {
    type Inner = T::Inner;
    fn map_assign_inner<F: FnMut(&mut Self::Inner) + Copy>(&mut self, f: F) {
        for i in 0..M {
            self[i].map_assign_inner(f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_map() {
        assert_eq!([0.0f32, 1.0, 2.0].map(|v| v * 2.0), [0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_2d_map() {
        assert_eq!(
            [[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]].map_elems(|v| v * 2.0),
            [[0.0f32, 2.0, 4.0], [6.0, 8.0, 10.0]]
        );
    }

    #[test]
    fn test_3d_map() {
        assert_eq!(
            [
                [[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
            ]
            .map_elems(|v| v * 2.0),
            [
                [[0.0f32, 2.0, 4.0], [6.0, 8.0, 10.0]],
                [[12.0, 14.0, 16.0], [18.0, 20.0, 22.0]]
            ]
        );
    }

    #[test]
    fn test_1d_add() {
        assert_eq!([1.0, 2.0, 3.0].add(&[4.0, 5.0, 6.0]), [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_2d_add() {
        assert_eq!(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].add(&[[1.0; 3]; 2]),
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
        );
    }

    #[test]
    fn test_3d_add() {
        assert_eq!(
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
            ]
            .add(&[[[1.0; 3]; 2]; 2]),
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
            ]
        );
    }
}
