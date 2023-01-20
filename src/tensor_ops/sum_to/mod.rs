mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{gradients::Tape, shapes::*, tensor::*};

pub trait SumKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>;
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>;
}

/// Reduction along multiple axes using `sum`.
pub trait SumTo: HasErr + HasShape {
    /// Sum reduction. **Pytorch equivalent**: `t.sum(Ax)`
    ///
    /// Example reducing a single axis:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let t: Tensor<Rank2<2, 3>, _> = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
    /// let r = t.sum::<Rank1<2>, _>(); // or `sum::<_, Axis<1>>()`
    /// assert_eq!(r.array(), [6.0, -6.0]);
    /// ```
    ///
    /// Reducing multiple axes:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// # let t = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
    /// let r = t.sum::<Rank0, _>(); // or `sum::<_, Axes2<0, 1>>()`
    /// assert_eq!(r.array(), 0.0);
    /// ```
    fn sum<Dst: Shape, Ax: Axes>(self) -> Self::WithShape<Dst>
    where
        Self::Shape: ReduceShapeTo<Dst, Ax>,
    {
        self.try_sum().unwrap()
    }
    /// Fallible version of [SumTo::sum]
    fn try_sum<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ReduceShapeTo<Dst, Ax>;
}

impl<S: Shape, E: Dtype, D: SumKernel<E>, T: Tape<D>> SumTo for Tensor<S, E, D, T> {
    fn try_sum<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ReduceShapeTo<Dst, Ax>,
    {
        let dst: Dst = self.shape().reduced();
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.upgrade(inp.device.forward(dst, &inp.storage)?);
        let phantom_out = out.clone();
        tape.try_alloc_grad(&inp)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
            inp.device.backward(grad_inp, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;
    use crate::tests::{assert_close, TestDevice};

    #[test]
    fn test_sum_1d() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([1.0, 2.0, 3.0]);
        let r = t.trace().sum::<Rank0, _>();
        assert_eq!(r.array(), 6.0);
        // NOTE: .exp() to make sure its using result grad properly
        let g = r.exp().backward();
        assert_eq!(g.get(&t).array(), [403.4288; 3]);
    }

    #[test]
    fn test_sum_axis_0_2d() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r = t.trace().sum::<Rank1<3>, _>();
        assert_eq!(r.array(), [-1.0, 6.0, -3.0]);
        let g = r.exp().mean().backward();
        assert_close(
            &g.get(&t).array(),
            &[[0.12262648, 134.47627, 0.01659569]; 2],
        );
    }

    #[test]
    fn test_sum_axis_1_2d() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r = t.trace().sum::<Rank1<2>, _>();
        assert_eq!(r.array(), [6.0, -4.0]);
        let g = r.exp().mean().backward();
        assert_close(&g.get(&t).array(), &[[201.7144; 3], [0.00915782; 3]]);
    }

    #[test]
    fn test_sum_axes_3d_to_1d() {
        let dev: TestDevice = Default::default();
        let t = dev.sample::<Rank3<2, 3, 4>, _>(rand_distr::StandardNormal);
        let r = t.trace().sum::<Rank1<3>, _>();
        let r2 = t.trace().sum::<Rank2<3, 4>, _>().sum::<Rank1<3>, _>();
        assert_close(&r.array(), &r2.array());
        let g = r.sum().backward();
        let g2 = r2.sum().backward();
        assert_close(&g.get(&t).array(), &g2.get(&t).array());
    }

    #[test]
    fn test_sum_chunking() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([[1.0; 100]; 60]);
        let r = t.trace().sum::<Rank1<60>, _>();
        assert_eq!(r.array(), [100.0; 60]);
        // let g = r.sum().backward();
        // assert_close(&g.get(&t).array(), &g2.get(&t).array());
    }
}
