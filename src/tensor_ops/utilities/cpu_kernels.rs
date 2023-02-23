use super::ops::{BinaryKernel, UnaryKernel};
use crate::{
    shapes::{Dtype, Shape},
    tensor::cpu::{Cpu, LendingIterator, NdIndex, StridedArray},
};

pub trait UnaryDerivative<E> {
    fn f(&self, x: &E) -> E;
    fn df(&self, x: &E) -> E;
}

pub trait BinaryDerivative<E> {
    fn f(&self, x: &E, y: &E) -> E;
    fn dfdx(&self, x: &E, y: &E) -> E;
    fn dfdy(&self, x: &E, y: &E) -> E;
}

impl<E: Dtype, Op: UnaryDerivative<E>> UnaryKernel<Op, E> for Cpu {
    fn forward<S: Shape>(
        &self,
        op: Op,
        inp: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        let mut out: Self::Storage<S, E> = inp.clone();
        // NOTE: we can iterate over buf here because we know inp & out
        // have exact same strides due to clone.
        for x in out.buf_iter_mut() {
            *x = op.f(x);
        }
        Ok(out)
    }

    fn backward<S: Shape>(
        &self,
        op: Op,
        inp: &Self::Storage<S, E>,
        grad_inp: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(grad_inp.data.len(), grad_out.data.len());
        debug_assert_eq!(inp.data.len(), grad_out.data.len());
        for (i, x) in grad_inp.buf_iter_mut().enumerate() {
            *x += op.df(&inp.data[i]) * grad_out.data[i];
        }
        Ok(())
    }
}

impl<E: Dtype, Op: BinaryDerivative<E>> BinaryKernel<Op, E> for Cpu {
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: &Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        let mut out: Self::Storage<S, E> = StridedArray::new(lhs.shape)?;

        let mut lhs_iter = lhs.iter();
        let mut rhs_iter = rhs.iter();
        // NOTE: we can use buf_iter_mut() here because StridedArray::new makes a contiguous array
        for o in out.buf_iter_mut() {
            let l = lhs_iter.next().unwrap();
            let r = rhs_iter.next().unwrap();
            *o = op.f(l, r);
        }
        Ok(out)
    }
    fn backward<S: Shape>(
        &self,
        op: Op,
        lhs: &Self::Storage<S, E>,
        grad_lhs: &mut Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
        grad_rhs: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        let mut lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
        let mut rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
        let lhs_buf = lhs.data.as_ref();
        let rhs_buf = rhs.data.as_ref();
        let grad_lhs_buf = std::sync::Arc::make_mut(&mut grad_lhs.data);
        let grad_rhs_buf = std::sync::Arc::make_mut(&mut grad_rhs.data);
        // NOTE: we can use .buf_iter() here because we know the outcome of this op is
        // contiguous from forward
        for &go in grad_out.buf_iter() {
            let lhs_i = lhs_idx.next().unwrap();
            let rhs_i = rhs_idx.next().unwrap();
            let l = &lhs_buf[lhs_i];
            let r = &rhs_buf[rhs_i];
            grad_lhs_buf[lhs_i] += op.dfdx(l, r) * go;
            grad_rhs_buf[rhs_i] += op.dfdy(l, r) * go;
        }
        Ok(())
    }
}
