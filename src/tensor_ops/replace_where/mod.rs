mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{
    gradients::{Merge, Tape},
    prelude::{DeviceStorage, HasErr, PutTape, SplitTape, Tensor},
    shapes::{Dtype, Shape},
};

trait ReplaceWhereKernel<E: Dtype>: DeviceStorage {
    fn forward<S: Shape>(
        &self,
        lhs: &Self::Storage<S, E>,
        cond: &Self::Storage<S, bool>,
        rhs: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err>;

    fn backward<S: Shape>(
        &self,
        grad_lhs: &mut Self::Storage<S, E>,
        cond: &Self::Storage<S, bool>,
        grad_rhs: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err>;
}

/// Choose values from two tensors using a boolean mask. Equivalent to `torch.where` from pytorch.
trait ReplaceWhere<Rhs>: HasErr {
    type Cond;
    type Output;

    /// Replace each value in self with the corresponding value in rhs if the corresponding value
    /// in cond is true.
    fn replace_where(self, cond: Self::Cond, rhs: Rhs) -> Self::Output {
        self.try_replace_where(cond, rhs).unwrap()
    }

    /// Fallible version of replace_where
    fn try_replace_where(self, cond: Self::Cond, rhs: Rhs) -> Result<Self::Output, Self::Err>;
}

impl<
        S: Shape,
        E: Dtype,
        D: ReplaceWhereKernel<E>,
        LhsTape: Tape<D> + Merge<RhsTape>,
        RhsTape: Tape<D>,
    > ReplaceWhere<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
{
    type Cond = Tensor<S, bool, D>;
    type Output = Tensor<S, E, D, LhsTape>;

    fn try_replace_where(
        self,
        cond: Self::Cond,
        rhs: Tensor<S, E, D, RhsTape>,
    ) -> Result<Self::Output, Self::Err> {
        let (lhs, tape) = self.split_tape();
        let (rhs, rhs_tape) = rhs.split_tape();

        let storage = lhs
            .device
            .forward(&lhs.storage, &cond.storage, &rhs.storage)?;
        let out = lhs.device.upgrade(storage);
        let phantom_out = out.clone();

        let mut tape = tape.merge(rhs_tape);
        tape.try_alloc_grad(&lhs)?;
        tape.try_alloc_grad(&rhs)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_lhs, grad_rhs, grad_out) = grads.muts_and_ref(&lhs, &rhs, &phantom_out);
            lhs.device
                .backward(grad_lhs, &cond.storage, grad_rhs, grad_out)
        });

        Ok(out.put_tape(tape))
    }
}
