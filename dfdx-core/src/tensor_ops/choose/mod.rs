mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{
    shapes::{Dtype, HasShape, Shape},
    tensor::{HasErr, Merge, PutTape, SplitTape, Storage, Tape, Tensor},
};

pub trait ChooseKernel<E: Dtype>: Storage<E> + Storage<bool> {
    fn forward<S: Shape>(
        &self,
        cond: &Tensor<S, bool, Self>,
        lhs: &Tensor<S, E, Self>,
        rhs: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err>;

    fn backward<S: Shape>(
        &self,
        cond: &Tensor<S, bool, Self>,
        lhs: &Tensor<S, E, Self>,
        grad_lhs: &mut <Self as Storage<E>>::Vec,
        rhs: &Tensor<S, E, Self>,
        grad_rhs: &mut <Self as Storage<E>>::Vec,
        grad_out: &<Self as Storage<E>>::Vec,
    ) -> Result<(), Self::Err>;
}

/// Choose values from two tensors using a boolean mask. Equivalent to `torch.where` from pytorch.
///
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let cond: Tensor<Rank1<3>, bool, _> = dev.tensor([true, false, true]);
/// let a: Tensor<Rank1<3>, f32, _> = dev.tensor([1.0, 2.0, 3.0]);
/// let b: Tensor<Rank1<3>, f32, _> = dev.tensor([-1.0, -2.0, -3.0]);
/// let c = cond.choose(a, b);
/// assert_eq!(c.array(), [1.0, -2.0, 3.0]);
/// ```
pub trait ChooseFrom<Lhs, Rhs>: HasErr {
    type Output;

    /// Construct a new tensor, where the output tensor contains the elements of lhs where self is
    /// true, and rhs where self is false.
    fn choose(self, lhs: Lhs, rhs: Rhs) -> Self::Output {
        self.try_choose(lhs, rhs).unwrap()
    }

    /// Fallible version of choose
    fn try_choose(self, lhs: Lhs, rhs: Rhs) -> Result<Self::Output, Self::Err>;
}

impl<
        S: Shape,
        E: Dtype,
        D: ChooseKernel<E>,
        LhsTape: Tape<E, D> + Merge<RhsTape>,
        RhsTape: Tape<E, D>,
    > ChooseFrom<Tensor<S, E, D, LhsTape>, Tensor<S, E, D, RhsTape>> for Tensor<S, bool, D>
{
    type Output = Tensor<S, E, D, LhsTape>;

    fn try_choose(
        self,
        lhs: Tensor<S, E, D, LhsTape>,
        rhs: Tensor<S, E, D, RhsTape>,
    ) -> Result<Self::Output, Self::Err> {
        assert_eq!(self.shape(), lhs.shape());
        assert_eq!(lhs.shape(), rhs.shape());

        let (lhs, tape) = lhs.split_tape();
        let (rhs, rhs_tape) = rhs.split_tape();

        let out = lhs.device.forward(&self, &lhs, &rhs)?;

        let lhs_ghost = lhs.ghost();
        let rhs_ghost = rhs.ghost();
        let out_ghost = out.ghost();
        let mut tape = tape.merge(rhs_tape);
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&lhs_ghost)?;
            grads.try_alloc_for(&rhs_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_lhs, grad_rhs, grad_out) =
                grads.muts_and_ref(&lhs_ghost, &rhs_ghost, &out_ghost);
            lhs.device
                .backward(&self, &lhs, grad_lhs, &rhs, grad_rhs, grad_out)
        });

        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::*;
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::TestDevice;

    #[test]
    fn test_choose_1d_backward() {
        let dev: TestDevice = Default::default();
        let cond = dev.tensor([false, true, false, true, false]);
        let a: Tensor<Rank1<5>, f32, _> = dev.sample_normal();
        let b: Tensor<Rank1<5>, f32, _> = dev.sample_normal();
        let r = cond.choose(a.leaky_trace(), b.leaky_trace());

        let a_array = a.array();
        let b_array = b.array();
        assert_eq!(
            r.array(),
            [b_array[0], a_array[1], b_array[2], a_array[3], b_array[4]]
        );
        let g = r.exp().sum().backward();
        assert_eq!(
            g.get(&a).array(),
            [0.0, a_array[1].exp(), 0.0, a_array[3].exp(), 0.0]
        );
        assert_eq!(
            g.get(&b).array(),
            [
                b_array[0].exp(),
                0.0,
                b_array[2].exp(),
                0.0,
                b_array[4].exp()
            ]
        );
    }

    #[test]
    fn test_choose_2d_backward() {
        let dev: TestDevice = Default::default();
        let cond = dev.tensor([[false, true], [true, false]]);
        let a: Tensor<_, f32, _> = dev.sample_normal();
        let b: Tensor<_, f32, _> = dev.sample_normal();
        let r = cond.choose(a.leaky_trace(), b.leaky_trace());

        let a_array = a.array();
        let b_array = b.array();
        assert_eq!(
            r.array(),
            [
                [b_array[0][0], a_array[0][1]],
                [a_array[1][0], b_array[1][1]]
            ]
        );
        let g = r.exp().sum().backward();
        assert_eq!(
            g.get(&a).array(),
            [[0.0, a_array[0][1].exp()], [a_array[1][0].exp(), 0.0]]
        );
        assert_eq!(
            g.get(&b).array(),
            [[b_array[0][0].exp(), 0.0], [0.0, b_array[1][1].exp()]]
        );
    }
}
