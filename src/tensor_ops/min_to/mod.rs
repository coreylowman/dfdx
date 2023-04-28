mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{shapes::*, tensor::*};

pub trait MinReduceKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Tensor<Src, E, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>;
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<Dst, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>;
}

/// Reduction along multiple axes using `min`.
pub trait MinTo: HasErr + HasShape {
    /// Min reduction. **Pytorch equivalent**: `t.amin(Ax)`
    ///
    /// **NOTE** This evenly distributes gradients between all equal maximum values, instead
    /// of only exactly 1 value.
    ///
    /// Example reducing a single axis:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let t: Tensor<Rank2<2, 3>, f32, _> = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
    /// let r = t.min::<Rank1<2>, _>(); // or `min::<_, Axis<1>>()`
    /// assert_eq!(r.array(), [1.0, -3.0]);
    /// ```
    ///
    /// Reducing multiple axes:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// # let t = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
    /// let r = t.min::<Rank0, _>();
    /// assert_eq!(r.array(), -3.0);
    /// ```
    fn min<Dst: Shape, Ax: Axes>(self) -> Self::WithShape<Dst>
    where
        Self::Shape: ReduceShapeTo<Dst, Ax>,
    {
        self.try_min().unwrap()
    }
    /// Fallible version of [MinTo::min]
    fn try_min<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ReduceShapeTo<Dst, Ax>;
}

impl<S: Shape, E: Dtype, D: MinReduceKernel<E>, T: Tape<E, D>> MinTo for Tensor<S, E, D, T> {
    fn try_min<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ReduceShapeTo<Dst, Ax>,
    {
        let dst: Dst = self.shape().reduced();
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.forward(dst, &inp)?;
        let inp_ghost = inp.ghost();
        let out_ghost = out.ghost();
        let out_clone = out.clone();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            inp.device.backward(&inp, grad_inp, &out_clone, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;
    use crate::tests::*;

    #[test]
    fn test_min_axis_0_2d() {
        let dev: TestDevice = Default::default();
        let t = dev
            .tensor([[1.0, 1.0, 2.0], [3.0, -2.0, 2.0]])
            .to_dtype::<TestDtype>();
        let r = t.leaky_trace().min::<Rank1<3>, _>();
        assert_close_to_literal!(r, [1.0, -2.0, 2.0]);
        let g = r.exp().mean().backward();
        assert_close_to_literal!(
            g.get(&t),
            [[0.90609396, 0.0, 2.463019], [0.0, 0.04511176, 2.463019]]
        );
    }

    #[test]
    fn test_min_axis_1_2d() {
        let dev: TestDevice = Default::default();
        let t = dev
            .tensor([[1.0, 1.0, 2.0], [3.0, -2.0, 2.0]])
            .to_dtype::<TestDtype>();
        let r = t.leaky_trace().min::<Rank1<2>, _>();
        assert_close_to_literal!(r, [1.0, -2.0]);
        let g = r.sum().backward();
        assert_close_to_literal!(g.get(&t), [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]);
    }

    #[test]
    fn test_min_axes_3d_to_1d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank3<2, 3, 4>, TestDtype, _> = dev.sample_normal();
        let r = t.leaky_trace().min::<Rank1<4>, _>();
        let r2 = t.leaky_trace().min::<Rank2<3, 4>, _>().min::<Rank1<4>, _>();
        assert_close_to_tensor!(r, r2);
        let g = r.mean().backward();
        let g2 = r2.mean().backward();
        assert_close_to_tensor!(g.get(&t), g2.get(&t));
    }

    #[test]
    fn test_min_negative_zero() {
        let dev: TestDevice = Default::default();
        let t = dev
            .tensor([[-0.0, 0.0], [0.0, -0.0], [-1.0, -0.0], [-1.0, 0.0]])
            .to_dtype::<TestDtype>();
        let r = t.leaky_trace().min::<_, Axis<1>>();
        assert_close_to_literal!(r, [-0.0, -0.0, -1.0, -1.0]);
        let g = r.sum().backward();
        assert_close_to_literal!(g.get(&t), [[1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]);
    }
}
