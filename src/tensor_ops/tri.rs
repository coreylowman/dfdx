use crate::shapes::{Dtype, Shape};
use crate::tensor::{HasErr, Tape, Tensor, TriangleTensor};

use super::TryMul;

/// Applies a 2D lower triangular mask by setting values above the diagonal to `E::default()`.
///
/// See [`TriangleTensor::lower_tri`].
pub fn lower_tri<S: Shape, E: Dtype, D: TriangleTensor<E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
    diagonal: impl Into<Option<isize>>,
) -> Tensor<S, E, D, T>
where
    Tensor<S, E, D, T>: TryMul<Tensor<S, E, D>> + HasErr<Err = D::Err>,
{
    t.lower_tri(diagonal)
}

/// Applies a 2D upper triangular mask by setting values below the diagonal to `E::default()`.
///
/// See [`TriangleTensor::upper_tri`].
pub fn upper_tri<S: Shape, E: Dtype, D: TriangleTensor<E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
    diagonal: impl Into<Option<isize>>,
) -> Tensor<S, E, D, T>
where
    Tensor<S, E, D, T>: TryMul<Tensor<S, E, D>> + HasErr<Err = D::Err>,
{
    t.upper_tri(diagonal)
}

impl<S: Shape, E: Dtype, D: TriangleTensor<E>, T: Tape<E, D>> Tensor<S, E, D, T>
where
    Self: TryMul<Tensor<S, E, D>> + HasErr<Err = D::Err>,
{
    /// See [lower_tri]
    pub fn try_lower_tri(
        self,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Self, <Self as HasErr>::Err> {
        let out = self
            .device
            .try_lower_tri_like(&self.shape, E::ONE, diagonal)?;
        self.try_mul(out)
    }

    /// See [lower_tri]
    pub fn lower_tri(self, diagonal: impl Into<Option<isize>>) -> Self {
        self.try_lower_tri(diagonal).unwrap()
    }

    /// See [upper_tri]
    pub fn try_upper_tri(
        self,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Self, <Self as HasErr>::Err> {
        let out = self
            .device
            .try_upper_tri_like(&self.shape, E::ONE, diagonal)?;
        self.try_mul(out)
    }

    /// See [upper_tri]
    pub fn upper_tri(self, diagonal: impl Into<Option<isize>>) -> Self {
        self.try_upper_tri(diagonal).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tests::*};

    #[test]
    fn test_tri() {
        let dev: TestDevice = Default::default();

        let t: Tensor<_, TestDtype, _> = dev.tensor(
            [[[
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
            ]; 4]; 3],
        );
        assert_close_to_literal!(
            t.clone().lower_tri(None),
            [[[
                [1., 0., 0., 0., 0., 0.],
                [1., 2., 0., 0., 0., 0.],
                [1., 2., 3., 0., 0., 0.],
                [1., 2., 3., 4., 0., 0.],
                [1., 2., 3., 4., 5., 0.],
            ]; 4]; 3]
        );
        assert_close_to_literal!(
            t.clone().lower_tri(2),
            [[[
                [1., 2., 3., 0., 0., 0.],
                [1., 2., 3., 4., 0., 0.],
                [1., 2., 3., 4., 5., 0.],
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
            ]; 4]; 3]
        );
        assert_close_to_literal!(
            t.upper_tri(-1),
            [[[
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
                [0., 2., 3., 4., 5., 6.],
                [0., 0., 3., 4., 5., 6.],
                [0., 0., 0., 4., 5., 6.],
            ]; 4]; 3]
        );
    }
}
