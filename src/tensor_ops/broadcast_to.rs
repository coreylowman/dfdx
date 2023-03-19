use crate::{shapes::*, tensor::*};

/// Broadcast self into a new shape.
pub trait BroadcastTo: HasErr + HasShape {
    /// Broadcast into shape `Dst` along axes `Ax`:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<3, 7>, f32, _> = dev.zeros();
    ///
    /// // broadcast axis 1
    /// let _ = a.clone().broadcast::<Rank3<3, 5, 7>, _>();
    ///
    /// // broadcast axis 0 and axis 2
    /// let _ = a.clone().broadcast::<Rank4<1, 3, 5, 7>, _>();
    /// ```
    fn broadcast<Dst: ConstShape, Ax: Axes>(self) -> Self::WithShape<Dst>
    where
        Self::Shape: BroadcastShapeTo<Dst, Ax>,
    {
        self.try_broadcast_like(&Default::default()).unwrap()
    }
    /// Fallible version of [BroadcastTo::broadcast]
    fn try_broadcast<Dst: ConstShape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: BroadcastShapeTo<Dst, Ax>,
    {
        self.try_broadcast_like(&Default::default())
    }
    /// Same as [BroadcastTo::broadcast], but the target shape is given
    fn broadcast_like<Dst: Shape, Ax: Axes>(self, dst: &Dst) -> Self::WithShape<Dst>
    where
        Self::Shape: BroadcastShapeTo<Dst, Ax>,
    {
        self.try_broadcast_like(dst).unwrap()
    }
    /// fallible version of [BroadcastTo::broadcast_like]
    fn try_broadcast_like<Dst: Shape, Ax: Axes>(
        self,
        dst: &Dst,
    ) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: BroadcastShapeTo<Dst, Ax>;
}

impl<S: Shape, E: Unit, D: DeviceStorage, T: Tape<E, D>> BroadcastTo for Tensor<S, E, D, T> {
    fn try_broadcast_like<Dst: Shape, Ax: Axes>(
        self,
        dst: &Dst,
    ) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: BroadcastShapeTo<Dst, Ax>,
    {
        self.shape().check(dst);

        Ok(Tensor {
            id: self.id,
            data: self.data,
            shape: *dst,
            strides: self.shape.broadcast_strides(self.strides),
            device: self.device,
            tape: self.tape,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;
    use crate::tests::*;

    #[test]
    #[should_panic]
    fn test_broadcast_incorrect_dims() {
        let dev: TestDevice = Default::default();
        let a: Tensor<(usize,), TestDtype, _> = dev.zeros_like(&(5,));
        let _: Tensor<(Const<3>, usize), TestDtype, _> = a.broadcast_like(&(Const, 7));
    }

    #[test]
    fn test_valid_1d_broadcasts() {
        let dev: TestDevice = Default::default();
        let _: Tensor<Rank1<5>, TestDtype, _> = dev.zeros::<Rank0>().broadcast();
        let _: Tensor<Rank2<5, 3>, TestDtype, _> = dev.zeros::<Rank1<3>>().broadcast();
        let _: Tensor<Rank2<5, 3>, TestDtype, _> = dev.zeros::<Rank1<5>>().broadcast();
        let _: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.zeros::<Rank2<5, 7>>().broadcast();
        let _: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.zeros::<Rank2<3, 7>>().broadcast();
        let _: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.zeros::<Rank2<3, 5>>().broadcast();
        let _: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.zeros::<Rank2<3, 5>>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank3<5, 7, 9>>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank3<3, 7, 9>>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank3<3, 5, 9>>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank3<3, 5, 7>>().broadcast();
    }

    #[test]
    fn test_valid_2d_broadcasts() {
        let dev: TestDevice = Default::default();
        let _: Tensor<Rank2<5, 3>, TestDtype, _> = dev.zeros::<Rank0>().broadcast();
        let _: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.zeros::<Rank1<3>>().broadcast();
        let _: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.zeros::<Rank1<5>>().broadcast();
        let _: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.zeros::<Rank1<7>>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank2<3, 5>>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank2<3, 7>>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank2<3, 9>>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank2<5, 7>>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank2<5, 9>>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank2<7, 9>>().broadcast();
    }

    #[test]
    fn test_valid_3d_broadcasts() {
        let dev: TestDevice = Default::default();
        let _: Tensor<Rank3<3, 5, 7>, TestDtype, _> = dev.zeros::<Rank0>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank1<3>>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank1<5>>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank1<7>>().broadcast();
        let _: Tensor<Rank4<3, 5, 7, 9>, TestDtype, _> = dev.zeros::<Rank1<9>>().broadcast();
    }

    #[test]
    fn test_broadcast_backwards() {
        let dev: TestDevice = Default::default();
        let a: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
        let b: Tensor<Rank2<5, 3>, TestDtype, _> = dev.sample_normal();
        let a_up = a.leaky_trace().broadcast::<Rank2<5, 3>, _>();
        a_up.array().assert_close(&[a.array(); 5], 1e-4);
        let r = a_up * b.clone();
        let g = r.exp().mean().backward();

        let a_up = a.clone().broadcast::<Rank2<5, 3>, _>();
        // a's gradient: (b * (b * a).exp()).sum(0) / 15
        let a_grad = (b.clone() * (b.clone() * a_up.clone()).exp()).sum::<Rank1<3>, _>() / 15.0;
        // b's gradient: (a * (b * a).exp()) / 15
        let b_grad = (a_up.clone() * (b.clone() * a_up).exp()) / 15.0;
        g.get(&a).array().assert_close(&a_grad.array(), 1e-4);
        g.get(&b).array().assert_close(&b_grad.array(), 1e-4);
    }

    #[test]
    fn test_broadcast_summed() {
        let dev: TestDevice = Default::default();
        let a: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
        let g = a
            .leaky_trace()
            .broadcast::<Rank2<4, 3>, _>()
            .exp()
            .mean()
            .backward();
        assert_close(&g.get(&a).array(), &a.array().map(|x| x.exp() / 3.0));
    }
}
