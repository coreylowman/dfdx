use crate::{
    shapes::{Dtype, Shape},
    tensor::{
        cpu::{LendingIterator, NdIndex},
        Cpu, Error, Storage, Tensor, ZerosTensor,
    },
};

impl<E: Dtype> super::ChooseKernel<E> for Cpu {
    fn forward<S: Shape>(
        &self,
        cond: &Tensor<S, bool, Self>,
        lhs: &Tensor<S, E, Self>,
        rhs: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        let mut out = self.try_zeros_like(&lhs.shape)?;
        let mut cond_iter = cond.iter();
        let mut lhs_iter = lhs.iter();
        let mut rhs_iter = rhs.iter();
        let mut out_iter = out.iter_mut();
        while let Some(((o, c), (l, r))) = out_iter
            .next()
            .zip(cond_iter.next())
            .zip(lhs_iter.next().zip(rhs_iter.next()))
        {
            *o = if *c { *l } else { *r };
        }
        Ok(out)
    }

    fn backward<S: Shape>(
        &self,
        cond: &Tensor<S, bool, Self>,
        lhs: &Tensor<S, E, Self>,
        grad_lhs: &mut <Self as Storage<E>>::Vec,
        rhs: &Tensor<S, E, Self>,
        grad_rhs: &mut <Self as Storage<E>>::Vec,
        grad_out: &<Self as Storage<E>>::Vec,
    ) -> Result<(), Error> {
        let mut lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
        let mut rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
        let mut out_idx = NdIndex::new(lhs.shape, lhs.shape.strides());
        let mut cond_iter = cond.iter();
        while let Some(((l, r), (o, c))) = lhs_idx
            .next()
            .zip(rhs_idx.next())
            .zip(out_idx.next().zip(cond_iter.next()))
        {
            if *c {
                grad_lhs[l] += grad_out[o];
            } else {
                grad_rhs[r] += grad_out[o];
            }
        }
        Ok(())
    }
}
