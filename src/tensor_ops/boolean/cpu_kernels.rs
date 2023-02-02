use crate::{
    prelude::{
        cpu::{LendingIterator, StridedArray},
        Cpu, HasErr,
    },
    shapes::{Shape, Unit},
};

use super::BooleanKernel;

impl Cpu {
    fn eval_binary<S: Shape, E: Unit, O: Fn(E, E) -> E>(
        &self,
        op: O,
        lhs: &StridedArray<S, E>,
        rhs: &StridedArray<S, E>,
    ) -> Result<StridedArray<S, E>, <Self as HasErr>::Err> {
        let mut out: StridedArray<S, E> = StridedArray::new(lhs.shape)?;
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
        inp: &Self::Storage<S, bool>,
    ) -> Result<Self::Storage<S, bool>, Self::Err> {
        let mut out: Self::Storage<S, bool> = inp.clone();
        for x in out.buf_iter_mut() {
            *x = !*x;
        }
        Ok(out)
    }

    fn and<S: Shape>(
        &self,
        lhs: &Self::Storage<S, bool>,
        rhs: &Self::Storage<S, bool>,
    ) -> Result<Self::Storage<S, bool>, Self::Err> {
        self.eval_binary(|l, r| l && r, lhs, rhs)
    }

    fn or<S: Shape>(
        &self,
        lhs: &Self::Storage<S, bool>,
        rhs: &Self::Storage<S, bool>,
    ) -> Result<Self::Storage<S, bool>, Self::Err> {
        self.eval_binary(|l, r| l || r, lhs, rhs)
    }

    fn xor<S: Shape>(
        &self,
        lhs: &Self::Storage<S, bool>,
        rhs: &Self::Storage<S, bool>,
    ) -> Result<Self::Storage<S, bool>, Self::Err> {
        self.eval_binary(|l, r| l ^ r, lhs, rhs)
    }
}
