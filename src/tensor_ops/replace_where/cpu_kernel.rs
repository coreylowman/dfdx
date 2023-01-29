use super::ReplaceWhereKernel;
use crate::{
    prelude::{
        cpu::{LendingIterator, StridedArray},
        Cpu, Dtype,
    },
    shapes::Shape,
};

impl<E: Dtype> ReplaceWhereKernel<E> for Cpu {
    fn forward<S: Shape>(
        &self,
        lhs: &Self::Storage<S, E>,
        cond: &Self::Storage<S, bool>,
        rhs: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        let mut out: Self::Storage<S, E> = StridedArray::new(lhs.shape)?;
        let mut cond_iter = cond.iter();
        let mut lhs_iter = lhs.iter();
        let mut rhs_iter = rhs.iter();
        let mut out_iter = out.iter_mut();
        while let Some(((o, c), (l, r))) = out_iter
            .next()
            .zip(cond_iter.next())
            .zip(lhs_iter.next().zip(rhs_iter.next()))
        {
            *o = if *c { *r } else { *l };
        }
        Ok(out)
    }

    fn backward<S: Shape>(
        &self,
        grad_lhs: &mut Self::Storage<S, E>,
        cond: &Self::Storage<S, bool>,
        grad_rhs: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        let mut cond_iter = cond.iter();
        let mut lhs_iter = grad_lhs.iter_mut();
        let mut rhs_iter = grad_rhs.iter_mut();
        let mut out_iter = grad_out.iter();
        while let Some(((l, r), (o, c))) = lhs_iter
            .next()
            .zip(rhs_iter.next())
            .zip(out_iter.next().zip(cond_iter.next()))
        {
            if *c {
                *r += *o;
            } else {
                *l += *o;
            }
        }
        Ok(())
    }
}
