use crate::{shapes::*, tensor::*};

/// Changes order of dimensions/axes
pub trait RealizeTo: HasErr + HasShape {
    /// Realizes the concrete shape of the tensor as another compatable shape:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
    /// let a = a.realize::<(usize, usize)>().unwrap();
    /// let a = a.realize::<Rank2<2, 3>>().unwrap();
    /// ```
    fn realize<Dst: Shape>(self) -> Option<Self::WithShape<Dst>>
    where
        Self::Shape: RealizeShapeTo<Dst>,
    {
        self.try_realize().unwrap()
    }

    /// Fallible version of [RealizeTo::realize].
    fn try_realize<Dst: Shape>(self) -> Result<Option<Self::WithShape<Dst>>, Self::Err>
    where
        Self::Shape: RealizeShapeTo<Dst>;
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T: Tape<E, D>> RealizeTo for Tensor<S, E, D, T> {
    fn try_realize<Dst: Shape>(self) -> Result<Option<Self::WithShape<Dst>>, Self::Err>
    where
        Self::Shape: RealizeShapeTo<Dst>,
    {
        Ok(self.shape.realized().map(|dst_shape| Tensor {
            id: self.id,
            data: self.data,
            strides: dst_shape.strides(),
            shape: dst_shape,
            device: self.device,
            tape: self.tape,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor_ops::*, tests::*};

    #[test]
    fn test_realize_2d() {
        let dev: TestDevice = Default::default();
        let src: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
        let dst: Tensor<(Const<2>, usize), TestDtype, _> =
            src.clone().realize::<(Const<2>, usize)>().unwrap();
        assert_eq!(src.as_vec(), dst.as_vec());
        let src = dst;
        let dst: Tensor<(usize, Const<3>), TestDtype, _> =
            src.clone().realize::<(usize, Const<3>)>().unwrap();
        assert_eq!(src.as_vec(), dst.as_vec());
        let src = dst;
        let dst: Tensor<(usize, usize), TestDtype, _> =
            src.clone().realize::<(usize, usize)>().unwrap();
        assert_eq!(src.as_vec(), dst.as_vec());
        assert!(src.clone().realize::<(usize, Const<4>)>().is_none());
        assert!(src.clone().realize::<(Const<1>, usize)>().is_none());
        assert!(src.clone().realize::<(Const<2>, Const<4>)>().is_none());
        assert!(src.clone().realize::<(Const<3>, Const<2>)>().is_none());
    }

    #[test]
    fn test_realize_3d() {
        let dev: TestDevice = Default::default();
        let src: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.sample_normal();
        let dst: Tensor<(Const<3>, usize, Const<7>), TestDtype, _> = src
            .clone()
            .realize::<(Const<3>, usize, Const<7>)>()
            .unwrap();
        assert_eq!(src.as_vec(), dst.as_vec());
        let src = dst;
        let dst: Tensor<(usize, Const<5>, usize), TestDtype, _> =
            src.clone().realize::<(usize, Const<5>, usize)>().unwrap();
        assert_eq!(src.as_vec(), dst.as_vec());
        let src = dst;
        let dst: Tensor<(usize, usize, usize), TestDtype, _> =
            src.clone().realize::<(usize, usize, usize)>().unwrap();
        assert_eq!(src.as_vec(), dst.as_vec());
        assert!(src.clone().realize::<(usize, Const<2>, usize)>().is_none());
        assert!(src
            .clone()
            .realize::<(Const<3>, Const<1>, Const<7>)>()
            .is_none());
        assert!(src.clone().realize::<(usize, usize, Const<3>)>().is_none());
    }

    #[test]
    fn test_realize_4d() {
        let dev: TestDevice = Default::default();
        let src: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.sample_normal();
        let dst: Tensor<(Const<3>, usize, Const<7>, usize), TestDtype, _> = src
            .clone()
            .realize::<(Const<3>, usize, Const<7>, usize)>()
            .unwrap();
        assert_eq!(src.as_vec(), dst.as_vec());
        let src = dst;
        let dst: Tensor<(usize, usize, usize, usize), TestDtype, _> = src
            .clone()
            .realize::<(usize, usize, usize, usize)>()
            .unwrap();
        assert_eq!(src.as_vec(), dst.as_vec());
        let src = dst;
        let dst: Tensor<(usize, Const<5>, Const<7>, Const<9>), TestDtype, _> = src
            .clone()
            .realize::<(usize, Const<5>, Const<7>, Const<9>)>()
            .unwrap();
        assert_eq!(src.as_vec(), dst.as_vec());
        assert!(src
            .clone()
            .realize::<(usize, Const<2>, usize, Const<9>)>()
            .is_none());
        assert!(src
            .clone()
            .realize::<(Const<3>, Const<1>, Const<7>, Const<9>)>()
            .is_none());
        assert!(src
            .clone()
            .realize::<(usize, usize, Const<3>, usize)>()
            .is_none());
    }

    #[test]
    fn test_realize_2d_backwards() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank2<3, 5>, TestDtype, _> = dev.sample_normal();
        let g1 = t.trace().exp().sum().backward();
        let g2 = t
            .trace()
            .realize::<(usize, usize)>()
            .unwrap()
            .exp()
            .sum()
            .backward();
        assert_eq!(g1.get(&t).as_vec(), g2.get(&t).as_vec());
    }

    #[test]
    fn test_realize_3d_backwards() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank3<3, 6, 9>, TestDtype, _> = dev.sample_normal();
        let g1 = t.trace().exp().sum().backward();
        let g2 = t
            .trace()
            .realize::<(usize, usize, usize)>()
            .unwrap()
            .exp()
            .sum()
            .backward();
        assert_eq!(g1.get(&t).array(), g2.get(&t).array());
    }

    #[test]
    fn test_realize_4d_backwards() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank4<3, 6, 9, 11>, TestDtype, _> = dev.sample_normal();
        let g1 = t.trace().exp().sum().backward();
        let g2 = t
            .trace()
            .realize::<(usize, usize, usize, usize)>()
            .unwrap()
            .exp()
            .sum()
            .backward();
        assert_eq!(g1.get(&t).array(), g2.get(&t).array());
    }

    #[test]
    fn test_valid_realizations() {
        let dev: TestDevice = Default::default();

        let x: Tensor<Rank2<3, 5>, TestDtype, _> = dev.sample_normal();
        let x = x.realize::<(Const<3>, usize)>().unwrap();
        let x = x.realize::<(usize, Const<5>)>().unwrap();
        let _ = x.realize::<(usize, usize)>().unwrap();

        let x: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.sample_normal();
        let x = x.realize::<(Const<3>, Const<5>, usize)>().unwrap();
        let x = x.realize::<(Const<3>, usize, Const<7>)>().unwrap();
        let x = x.realize::<(usize, Const<5>, Const<7>)>().unwrap();
        let x = x.realize::<(Const<3>, usize, usize)>().unwrap();
        let x = x.realize::<(usize, Const<5>, usize)>().unwrap();
        let x = x.realize::<(usize, usize, Const<7>)>().unwrap();
        let _ = x.realize::<(usize, usize, usize)>().unwrap();

        let x: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.sample_normal();
        let x = x
            .realize::<(Const<3>, Const<5>, Const<7>, usize)>()
            .unwrap();
        let x = x
            .realize::<(Const<3>, Const<5>, usize, Const<9>)>()
            .unwrap();
        let x = x
            .realize::<(Const<3>, usize, Const<7>, Const<9>)>()
            .unwrap();
        let x = x
            .realize::<(usize, Const<5>, Const<7>, Const<9>)>()
            .unwrap();
        let x = x.realize::<(Const<3>, Const<5>, usize, usize)>().unwrap();
        let x = x.realize::<(Const<3>, usize, usize, Const<9>)>().unwrap();
        let x = x.realize::<(usize, usize, Const<7>, Const<9>)>().unwrap();
        let x = x.realize::<(Const<3>, usize, usize, usize)>().unwrap();
        let x = x.realize::<(usize, Const<5>, usize, usize)>().unwrap();
        let x = x.realize::<(usize, usize, Const<7>, usize)>().unwrap();
        let x = x.realize::<(usize, usize, usize, Const<9>)>().unwrap();
        let _ = x.realize::<(usize, usize, usize, usize)>().unwrap();
    }
}
