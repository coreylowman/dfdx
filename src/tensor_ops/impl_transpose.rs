use crate::prelude::*;

pub trait Transpose<const A: usize, const B: usize> {
    type Output;
    fn transpose(self) -> Self::Output;
}

// TEMPORARY IMPLEMENTATIONS
impl<const A: usize, const B: usize, const C: usize> Transpose<0, 1> for Tensor3D<A, B, C> {
    type Output = Tensor3D<B, A, C>;
    fn transpose(self) -> Self::Output {
        let mut new = Tensor3D::zeros();
        // Copy data
        let data = self.data();
        let new_data = new.mut_data();

        #[allow(clippy::needless_range_loop)]
        for i in 0..B {
            for j in 0..A {
                new_data[i][j] = data[j][i];
            }
        }
        new
    }
}

// Didn't feel like implementing these since the switch to flat mapped tensors will make them all obselete.
impl<const A: usize, const B: usize, const C: usize> Transpose<1, 2> for Tensor3D<A, B, C> {
    type Output = Tensor3D<A, C, B>;
    fn transpose(self) -> Self::Output {
        todo!()
    }
}

impl<const A: usize, const B: usize, const C: usize> Transpose<0, 2> for Tensor3D<A, B, C> {
    type Output = Tensor3D<C, B, A>;
    fn transpose(self) -> Self::Output {
        todo!()
    }
}

#[cfg(test)]
mod tests {

    use crate::{prelude::*, tests::assert_close};

    use super::Transpose;

    #[test]
    fn transpose_test() {
        let t: Tensor3D<2, 3, 1> = Tensor3D::new([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]);
        let y = Transpose::<0, 1>::transpose(t);
        assert_close(y.data(), &[[[1.0], [4.0]], [[2.0], [5.0]], [[3.0], [6.0]]]);
    }
}
