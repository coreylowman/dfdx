use crate::{
    shapes::{Shape, Unit},
    tensor::{cpu::LendingIterator, Cpu, HasErr, Tensor, ZerosTensor},
};

use super::BooleanKernel;

impl Cpu {
    fn eval_binary<S: Shape, E: Unit, O: Fn(E, E) -> E>(
        &self,
        op: O,
        lhs: &Tensor<S, E, Self>,
        rhs: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, <Self as HasErr>::Err> {
        let mut out = self.try_zeros_like(&lhs.shape)?;
        let mut lhs_iter = lhs.iter();
        let mut rhs_iter = rhs.iter();
        let mut out_iter = out.iter_mut();
        while let Some((o, (l, r))) = out_iter.next().zip(lhs_iter.next().zip(rhs_iter.next())) {
            *o = op(*l, *r);
        }
        Ok(out)
    }
}

impl BooleanKernel for Cpu {
    fn not<S: Shape>(
        &self,
        inp: &Tensor<S, bool, Self>,
    ) -> Result<Tensor<S, bool, Self>, Self::Err> {
        let mut out = inp.clone();
        for x in out.buf_iter_mut() {
            *x = !*x;
        }
        Ok(out)
    }

    fn and<S: Shape>(
        &self,
        lhs: &Tensor<S, bool, Self>,
        rhs: &Tensor<S, bool, Self>,
    ) -> Result<Tensor<S, bool, Self>, Self::Err> {
        self.eval_binary(|l, r| l && r, lhs, rhs)
    }

    fn or<S: Shape>(
        &self,
        lhs: &Tensor<S, bool, Self>,
        rhs: &Tensor<S, bool, Self>,
    ) -> Result<Tensor<S, bool, Self>, Self::Err> {
        self.eval_binary(|l, r| l || r, lhs, rhs)
    }

    fn xor<S: Shape>(
        &self,
        lhs: &Tensor<S, bool, Self>,
        rhs: &Tensor<S, bool, Self>,
    ) -> Result<Tensor<S, bool, Self>, Self::Err> {
        self.eval_binary(|l, r| l ^ r, lhs, rhs)
    }
}
