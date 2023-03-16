use crate::{shapes::*, tensor::*};

/// Changes order of dimensions/axes
pub trait PermuteTo: HasErr + HasShape {
    /// Permutes the tensor:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank3<1, 2, 3>, f32, _> = dev.zeros();
    /// let _ = a.clone().permute::<Rank3<3, 2, 1>, _>();
    /// let _ = a.clone().permute::<_, Axes3<2, 1, 0>>();
    /// ```
    fn permute<Dst: Shape, Ax: Axes>(self) -> Self::WithShape<Dst>
    where
        Self::Shape: PermuteShapeTo<Dst, Ax>,
    {
        self.try_permute().unwrap()
    }
    /// Fallible version of [PermuteTo::permute]
    fn try_permute<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: PermuteShapeTo<Dst, Ax>;
}

impl<S: Shape, E: Unit, D: DeviceStorage, T: Tape<E, D>> PermuteTo for Tensor<S, E, D, T> {
    fn try_permute<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: PermuteShapeTo<Dst, Ax>,
    {
        Ok(Tensor {
            id: self.id,
            data: self.data,
            shape: self.shape.permuted(),
            strides: self.shape.permute_strides(self.strides),
            device: self.device,
            tape: self.tape,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use crate::{tensor_ops::*, tests::*};

    #[test]
    fn test_permute_2d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
        let r = t.clone().permute();
        let t_array = t.array();
        let r_array = r.array();
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(r_array[j][i], t_array[i][j]);
            }
        }
    }

    #[test]
    fn test_permute_3d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.sample_normal();
        let r = t.clone().permute::<Rank3<5, 7, 3>, _>();
        let t_array = t.array();
        let r_array = r.array();
        for i in 0..3 {
            for j in 0..5 {
                for k in 0..7 {
                    assert_eq!(r_array[j][k][i], t_array[i][j][k]);
                }
            }
        }
    }

    #[test]
    fn test_permute_4d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.sample_normal();
        let r = t.clone().permute::<Rank4<5, 9, 3, 7>, _>();
        let t_array = t.array();
        let r_array = r.array();
        for i in 0..3 {
            for j in 0..5 {
                for k in 0..7 {
                    for l in 0..9 {
                        assert_eq!(r_array[j][l][i][k], t_array[i][j][k][l]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_permute_2d_backwards() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank2<3, 5>, TestDtype, _> = dev.sample_normal();
        let g1 = t.leaky_trace().exp().sum().backward();
        let g2 = t.leaky_trace().permute().exp().sum().backward();
        assert_eq!(g1.get(&t).array(), g2.get(&t).array());
    }

    #[test]
    fn test_permute_3d_backwards() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank3<3, 6, 9>, TestDtype, _> = dev.sample_normal();
        let g1 = t.leaky_trace().exp().sum().backward();
        let g2 = t
            .leaky_trace()
            .permute::<Rank3<6, 3, 9>, _>()
            .exp()
            .sum()
            .backward();
        assert_eq!(g1.get(&t).array(), g2.get(&t).array());
    }

    #[test]
    fn test_permute_4d_backwards() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank4<3, 6, 9, 11>, TestDtype, _> = dev.sample_normal();
        let g1 = t.leaky_trace().exp().sum().backward();
        let g2 = t
            .leaky_trace()
            .permute::<Rank4<6, 3, 11, 9>, _>()
            .exp()
            .sum()
            .backward();
        assert_eq!(g1.get(&t).array(), g2.get(&t).array());
    }

    #[test]
    fn test_valid_permutations() {
        let dev: TestDevice = Default::default();

        let x: Tensor<Rank2<3, 5>, TestDtype, _> = dev.sample_normal();
        let _ = x.permute::<_, Axes2<1, 0>>();

        let x: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.sample_normal();
        let _ = x.clone().permute::<_, Axes3<0, 2, 1>>();
        let _ = x.clone().permute::<_, Axes3<1, 0, 2>>();
        let _ = x.clone().permute::<_, Axes3<1, 2, 0>>();
        let _ = x.clone().permute::<_, Axes3<2, 0, 1>>();
        let _ = x.permute::<_, Axes3<2, 1, 0>>();

        let x: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.sample_normal();
        x.clone().permute::<_, Axes4<0, 1, 2, 3>>();
        x.clone().permute::<_, Axes4<0, 1, 3, 2>>();
        x.clone().permute::<_, Axes4<0, 2, 1, 3>>();
        x.clone().permute::<_, Axes4<0, 2, 3, 1>>();
        x.clone().permute::<_, Axes4<0, 3, 2, 1>>();
        x.clone().permute::<_, Axes4<0, 3, 1, 2>>();
        x.clone().permute::<_, Axes4<1, 0, 2, 3>>();
        x.clone().permute::<_, Axes4<1, 0, 3, 2>>();
        x.clone().permute::<_, Axes4<1, 2, 0, 3>>();
        x.clone().permute::<_, Axes4<1, 2, 3, 0>>();
        x.clone().permute::<_, Axes4<1, 3, 0, 2>>();
        x.clone().permute::<_, Axes4<1, 3, 2, 0>>();
        x.clone().permute::<_, Axes4<2, 0, 1, 3>>();
        x.clone().permute::<_, Axes4<2, 0, 3, 1>>();
        x.clone().permute::<_, Axes4<2, 1, 0, 3>>();
        x.clone().permute::<_, Axes4<2, 1, 3, 0>>();
        x.clone().permute::<_, Axes4<2, 3, 0, 1>>();
        x.clone().permute::<_, Axes4<2, 3, 1, 0>>();
        x.clone().permute::<_, Axes4<3, 0, 1, 2>>();
        x.clone().permute::<_, Axes4<3, 0, 2, 1>>();
        x.clone().permute::<_, Axes4<3, 1, 0, 2>>();
        x.clone().permute::<_, Axes4<3, 1, 2, 0>>();
        x.clone().permute::<_, Axes4<3, 2, 0, 1>>();
        x.permute::<_, Axes4<3, 2, 1, 0>>();
    }
}
