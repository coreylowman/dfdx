use crate::shapes::{Dtype, Shape};
use crate::tensor::{HasErr, Tape, Tensor, TriangleTensor};

use super::TryMul;

pub fn lower_tri<S: Shape, E: Dtype, D: TriangleTensor<E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
    diagonal: impl Into<Option<isize>>,
) -> Tensor<S, E, D, T>
where
    Tensor<S, E, D, T>: TryMul<Tensor<S, E, D>> + HasErr<Err = D::Err>,
{
    t.lower_tri(diagonal)
}

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
    pub fn try_lower_tri(
        self,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Self, <Self as HasErr>::Err> {
        let out = self
            .device
            .try_lower_tri_like(&self.shape, E::one(), diagonal)?;
        self.try_mul(out)
    }

    pub fn lower_tri(self, diagonal: impl Into<Option<isize>>) -> Self {
        self.try_lower_tri(diagonal).unwrap()
    }

    pub fn try_upper_tri(
        self,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Self, <Self as HasErr>::Err> {
        let out = self
            .device
            .try_upper_tri_like(&self.shape, E::one(), diagonal)?;
        self.try_mul(out)
    }

    pub fn upper_tri(self, diagonal: impl Into<Option<isize>>) -> Self {
        self.try_upper_tri(diagonal).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        prelude::{AsArray, TensorFrom},
        tests::TestDevice,
    };

    #[test]
    fn test_tri() {
        let dev: TestDevice = Default::default();

        let t = dev.tensor(
            [[[
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
            ]; 4]; 3],
        );
        assert_eq!(
            t.clone().lower_tri(None).array(),
            [[[
                [1., 0., 0., 0., 0., 0.],
                [1., 2., 0., 0., 0., 0.],
                [1., 2., 3., 0., 0., 0.],
                [1., 2., 3., 4., 0., 0.],
                [1., 2., 3., 4., 5., 0.],
            ]; 4]; 3],
        );
        assert_eq!(
            t.clone().lower_tri(2).array(),
            [[[
                [1., 2., 3., 0., 0., 0.],
                [1., 2., 3., 4., 0., 0.],
                [1., 2., 3., 4., 5., 0.],
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
            ]; 4]; 3],
        );
        assert_eq!(
            t.upper_tri(-1).array(),
            [[[
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
                [0., 2., 3., 4., 5., 6.],
                [0., 0., 3., 4., 5., 6.],
                [0., 0., 0., 4., 5., 6.],
            ]; 4]; 3],
        );
    }
}
