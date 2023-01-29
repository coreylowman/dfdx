mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{
    gradients::{Merge, Tape},
    prelude::{DeviceStorage, HasErr, PutTape, SplitTape, Tensor},
    shapes::{Dtype, Shape},
};

pub trait ChooseKernel<E: Dtype>: DeviceStorage {
    fn forward<S: Shape>(
        &self,
        cond: &Self::Storage<S, bool>,
        lhs: &Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err>;

    fn backward<S: Shape>(
        &self,
        cond: &Self::Storage<S, bool>,
        grad_lhs: &mut Self::Storage<S, E>,
        grad_rhs: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err>;
}

/// Choose values from two tensors using a boolean mask. Equivalent to `torch.where` from pytorch.
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
        LhsTape: Tape<D> + Merge<RhsTape>,
        RhsTape: Tape<D>,
    > ChooseFrom<Tensor<S, E, D, LhsTape>, Tensor<S, E, D, RhsTape>> for Tensor<S, bool, D>
{
    type Output = Tensor<S, E, D, LhsTape>;

    fn try_choose(
        self,
        lhs: Tensor<S, E, D, LhsTape>,
        rhs: Tensor<S, E, D, RhsTape>,
    ) -> Result<Self::Output, Self::Err> {
        let (lhs, tape) = lhs.split_tape();
        let (rhs, rhs_tape) = rhs.split_tape();

        let storage = lhs
            .device
            .forward(&self.storage, &lhs.storage, &rhs.storage)?;
        let out = lhs.device.upgrade(storage);
        let phantom_out = out.clone();

        let mut tape = tape.merge(rhs_tape);
        tape.try_alloc_grad(&lhs)?;
        tape.try_alloc_grad(&rhs)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_lhs, grad_rhs, grad_out) = grads.muts_and_ref(&lhs, &rhs, &phantom_out);
            lhs.device
                .backward(&self.storage, grad_lhs, grad_rhs, grad_out)
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
        let r = cond.choose(a.trace(), b.trace());

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
        let r = cond.choose(a.trace(), b.trace());

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
