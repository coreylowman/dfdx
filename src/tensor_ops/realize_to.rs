use crate::{shapes::*, tensor::*};

/// Realizes the concrete shape of the tensor as another compatable shape,
/// or returns the original tensor if the new shape's dimensions are incompatable.
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
/// let a = a.realize::<(usize, usize)>();
/// let mut a = a.realize::<Rank2<2, 3>>();
/// match a.try_realize::<(usize, Const<4>)>() {
///     Ok(new) => println!("Shape was properly realized, returned new tensor"),
///     Err(old) => println!("Shape could not be realized, returned the original tensor"),
/// }
/// ```
pub trait RealizeTo: HasErr + HasShape {
    /// Realizes the concrete shape of the tensor as another compatable shape,
    /// or returns the original tensor if the new shape's dimensions are incompatable.
    fn realize<Dst: Shape<Concrete = <<Self as HasShape>::Shape as Shape>::Concrete>>(
        self,
    ) -> Self::WithShape<Dst>
    where
        Self::Shape: RealizeShapeTo<Dst>,
        Self: std::fmt::Debug,
    {
        self.try_realize::<Dst>().unwrap()
    }

    /// Realizes the concrete shape of the tensor as another compatable shape,
    /// or returns the original tensor if the new shape's dimensions are incompatable.
    fn try_realize<Dst: Shape<Concrete = <<Self as HasShape>::Shape as Shape>::Concrete>>(
        self,
    ) -> Result<Self::WithShape<Dst>, Self>
    where
        Self::Shape: RealizeShapeTo<Dst>;
}

impl<S: Shape, E, D: Storage<E>, T: Tape<E, D>> RealizeTo for Tensor<S, E, D, T> {
    fn try_realize<Dst: Shape<Concrete = S::Concrete>>(self) -> Result<Self::WithShape<Dst>, Self>
    where
        Self::Shape: RealizeShapeTo<Dst>,
    {
        if let Some(dst_shape) = self.shape.realized() {
            Ok(Tensor {
                id: self.id,
                data: self.data,
                strides: self.strides,
                shape: dst_shape,
                device: self.device,
                tape: self.tape,
            })
        } else {
            Err(self)
        }
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
        let dst = src.clone().realize::<(Const<2>, usize)>();
        assert_eq!(src.as_vec(), dst.as_vec());
        let src = dst;
        let dst = src.clone().realize::<(usize, Const<3>)>();
        assert_eq!(src.as_vec(), dst.as_vec());
        let mut src = dst;
        let dst: Tensor<(usize, usize), TestDtype, _> = src.clone().realize::<(usize, usize)>();
        assert_eq!(src.as_vec(), dst.as_vec());
        src = src.try_realize::<(usize, Const<4>)>().unwrap_err();
        src = src.try_realize::<(Const<1>, usize)>().unwrap_err();
        src = src.try_realize::<(Const<2>, Const<4>)>().unwrap_err();
        src = src.try_realize::<(Const<3>, Const<2>)>().unwrap_err();
        assert_eq!(src.as_vec(), dst.as_vec());
    }

    #[test]
    fn test_realize_3d() {
        let dev: TestDevice = Default::default();
        let src: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.sample_normal();
        let dst = src.clone().realize::<(Const<3>, usize, Const<7>)>();
        assert_eq!(src.as_vec(), dst.as_vec());
        let src = dst;
        let dst = src.clone().realize::<(usize, Const<5>, usize)>();
        assert_eq!(src.as_vec(), dst.as_vec());
        let mut src = dst;
        let dst = src.clone().realize::<(usize, usize, usize)>();
        assert_eq!(src.as_vec(), dst.as_vec());
        // Ensure we get back the original tensor on error
        src = src.try_realize::<(usize, Const<2>, usize)>().unwrap_err();
        src = src
            .try_realize::<(Const<3>, Const<1>, Const<7>)>()
            .unwrap_err();
        src = src.try_realize::<(usize, usize, Const<3>)>().unwrap_err();
        assert_eq!(src.as_vec(), dst.as_vec());
    }

    #[test]
    fn test_realize_4d() {
        let dev: TestDevice = Default::default();
        let src: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.sample_normal();
        let dst: Tensor<(Const<3>, usize, Const<7>, usize), TestDtype, _> = src
            .clone()
            .try_realize::<(Const<3>, usize, Const<7>, usize)>()
            .unwrap();
        assert_eq!(src.as_vec(), dst.as_vec());
        let src = dst;
        let dst: Tensor<(usize, usize, usize, usize), TestDtype, _> = src
            .clone()
            .try_realize::<(usize, usize, usize, usize)>()
            .unwrap();
        assert_eq!(src.as_vec(), dst.as_vec());
        let mut src = dst;
        let dst: Tensor<(usize, Const<5>, Const<7>, Const<9>), TestDtype, _> = src
            .clone()
            .try_realize::<(usize, Const<5>, Const<7>, Const<9>)>()
            .unwrap();
        assert_eq!(src.as_vec(), dst.as_vec());
        src = src
            .try_realize::<(usize, Const<2>, usize, Const<9>)>()
            .unwrap_err();
        src = src
            .try_realize::<(Const<3>, Const<1>, Const<7>, Const<9>)>()
            .unwrap_err();
        src = src
            .try_realize::<(usize, usize, Const<3>, usize)>()
            .unwrap_err();
        assert_eq!(src.as_vec(), dst.as_vec());
    }

    #[test]
    fn test_realize_2d_backwards() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank2<3, 5>, TestDtype, _> = dev.sample_normal();
        let g1 = t.leaky_trace().exp().sum().backward();
        let g2 = t
            .leaky_trace()
            .try_realize::<(usize, usize)>()
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
        let g1 = t.leaky_trace().exp().sum().backward();
        let g2 = t
            .leaky_trace()
            .try_realize::<(usize, usize, usize)>()
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
        let g1 = t.leaky_trace().exp().sum().backward();
        let g2 = t
            .leaky_trace()
            .try_realize::<(usize, usize, usize, usize)>()
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
        let x = x.try_realize::<(Const<3>, usize)>().unwrap();
        let x = x.try_realize::<(usize, Const<5>)>().unwrap();
        let _ = x.try_realize::<(usize, usize)>().unwrap();

        let x: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.sample_normal();
        let x = x.try_realize::<(Const<3>, Const<5>, usize)>().unwrap();
        let x = x.try_realize::<(Const<3>, usize, Const<7>)>().unwrap();
        let x = x.try_realize::<(usize, Const<5>, Const<7>)>().unwrap();
        let x = x.try_realize::<(Const<3>, usize, usize)>().unwrap();
        let x = x.try_realize::<(usize, Const<5>, usize)>().unwrap();
        let x = x.try_realize::<(usize, usize, Const<7>)>().unwrap();
        let _ = x.try_realize::<(usize, usize, usize)>().unwrap();

        let x: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.sample_normal();
        let x = x
            .try_realize::<(Const<3>, Const<5>, Const<7>, usize)>()
            .unwrap();
        let x = x
            .try_realize::<(Const<3>, Const<5>, usize, Const<9>)>()
            .unwrap();
        let x = x
            .try_realize::<(Const<3>, usize, Const<7>, Const<9>)>()
            .unwrap();
        let x = x
            .try_realize::<(usize, Const<5>, Const<7>, Const<9>)>()
            .unwrap();
        let x = x
            .try_realize::<(Const<3>, Const<5>, usize, usize)>()
            .unwrap();
        let x = x
            .try_realize::<(Const<3>, usize, usize, Const<9>)>()
            .unwrap();
        let x = x
            .try_realize::<(usize, usize, Const<7>, Const<9>)>()
            .unwrap();
        let x = x.try_realize::<(Const<3>, usize, usize, usize)>().unwrap();
        let x = x.try_realize::<(usize, Const<5>, usize, usize)>().unwrap();
        let x = x.try_realize::<(usize, usize, Const<7>, usize)>().unwrap();
        let x = x.try_realize::<(usize, usize, usize, Const<9>)>().unwrap();
        let _ = x.try_realize::<(usize, usize, usize, usize)>().unwrap();
    }
}
