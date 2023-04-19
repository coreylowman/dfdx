mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{shapes::*, tensor::*};

pub trait SumKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Tensor<Src, E, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>;
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &impl Tensorlike<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
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
    /// let t: Tensor<Rank2<2, 3>, f32, _> = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
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

impl<S: Shape, E: Dtype, D: SumKernel<E>, T: Tape<E, D>> SumTo for Tensor<S, E, D, T> {
    fn try_sum<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ReduceShapeTo<Dst, Ax>,
    {
        let dst: Dst = self.shape().reduced();
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.forward(dst, &inp)?;
        let inp_ghost = inp.ghost();
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            inp.device.backward(dst, &inp_ghost, grad_inp, grad_out)
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
    fn test_sum_1d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([1.0, 2.0, 3.0]);
        let r = t.leaky_trace().sum::<Rank0, _>();
        let e = 6.0f64;
        assert_close_to_literal!(r, e);
        // NOTE: .exp() to make sure its using result grad properly
        let g = r.exp().backward();
        assert_close_to_literal!(g.get(&t), [e.exp(); 3]);
    }

    #[test]
    fn test_sum_axis_0_2d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r = t.leaky_trace().sum::<Rank1<3>, _>();
        let e = [-1.0f64, 6.0, -3.0];
        assert_close_to_literal!(r, e);
        let g = r.exp().mean().backward();
        assert_close_to_literal!(g.get(&t), [e.map(|x| x.exp() / 3.0); 2], 1e-4);
    }

    #[test]
    fn test_sum_axis_1_2d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r = t.leaky_trace().sum::<Rank1<2>, _>();
        let e = [6.0f64, -4.0];
        assert_close_to_literal!(r, e);
        let g = r.exp().mean().backward();
        assert_close_to_literal!(g.get(&t), [[e[0].exp() / 2.0; 3], [e[1].exp() / 2.0; 3]]);
    }

    #[test]
    fn test_sum_axes_3d_to_1d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank3<2, 3, 4>, TestDtype, _> = dev.sample_normal();
        let r = t.leaky_trace().sum::<Rank1<3>, _>();
        let r2 = t.leaky_trace().sum::<Rank2<3, 4>, _>().sum::<Rank1<3>, _>();
        assert_close_to_tensor!(r, r2);
        let g = r.sum().backward();
        let g2 = r2.sum().backward();
        assert_close_to_tensor!(g.get(&t), g2.get(&t));
    }

    #[test]
    fn test_sum_broadcasted() {
        let dev: TestDevice = Default::default();
        let t1: Tensor<Rank2<4, 3>, TestDtype, _> = dev.sample_normal();
        let t2 = t1.clone().broadcast::<Rank3<4, 3, 5>, _>();
        let r1 = t1.leaky_trace().sum::<Rank1<4>, _>() * 5.0;
        let r2 = t2.leaky_trace().sum::<Rank1<4>, _>();
        assert_close_to_tensor!(r1, r2, 3e-6);
        let g = r1.sum().backward();
        assert_close_to_literal!(g.get(&t1), [[5.0; 3]; 4]);
    }

    #[test]
    fn test_sum_chunking() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([[1.0; 100]; 60]);
        let r = t.leaky_trace().sum::<Rank1<60>, _>();
        assert_close_to_literal!(r, [100.0; 60]);
        let g = r.sum().backward();
        assert_close_to_tensor!(g.get(&t), t);
    }

    #[test]
    fn test_sum_reduce_to_more_than_physical_elements() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([1.0, 2.0, 3.0]);
        let b = a.broadcast::<Rank3<4, 3, 2>, _>();
        let c = b.sum::<Rank2<4, 3>, _>();
        assert_close_to_literal!(c, [[2.0, 4.0, 6.0]; 4]);
    }

    #[test]
    fn test_sum_reduce_to_0d_from_broadcasted() {
        let dev: TestDevice = Default::default();
        let a: Tensor<Rank1<3>, TestDtype, _> = dev.ones();
        let b = a.leaky_trace().broadcast::<Rank3<4, 3, 2>, _>();
        let c = b.sum();
        assert_close_to_literal!(c, 24.0);
        let g = c.backward();
        assert_close_to_literal!(g.get(&a), [8.0; 3]);
    }
}
