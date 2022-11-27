use super::ops::{BinaryKernel, UnaryKernel};
use crate::{
    arrays::{Dtype, Shape},
    devices::{cpu::LendingIterator, Cpu, ZerosLike},
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
    fn unary_fwd<S: Shape>(
        &self,
        op: Op,
        inp: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        let mut out: Self::Storage<S, E> = inp.try_clone()?;
        for x in out.buf_iter_mut() {
            *x = op.f(x);
        }
        Ok(out)
    }

    fn unary_bwd<S: Shape>(
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
    fn binary_fwd<S: Shape>(
        &self,
        op: Op,
        lhs: &Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        let mut out: Self::Storage<S, E> = self.try_zeros_like(lhs.shape)?;
        let mut lhs_iter = lhs.iter();
        let mut rhs_iter = rhs.iter();
        let mut out_iter = out.iter_mut();
        while let Some((o, (l, r))) = out_iter.next().zip(lhs_iter.next().zip(rhs_iter.next())) {
            *o = op.f(l, r);
        }
        Ok(out)
    }
    fn binary_bwd<S: Shape>(
        &self,
        op: Op,
        lhs: &Self::Storage<S, E>,
        grad_lhs: &mut Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
        grad_rhs: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        let mut lhs_iter = lhs.iter();
        let mut rhs_iter = rhs.iter();
        let mut grad_lhs_iter = grad_lhs.iter_mut();
        let mut grad_rhs_iter = grad_rhs.iter_mut();
        let mut grad_out_iter = grad_out.iter();
        for _ in 0..lhs.shape.num_elements() {
            let l = lhs_iter.next().unwrap();
            let r = rhs_iter.next().unwrap();
            let go = *grad_out_iter.next().unwrap();
            let gl = grad_lhs_iter.next().unwrap();
            *gl += op.dfdx(l, r) * go;
            let gr = grad_rhs_iter.next().unwrap();
            *gr += op.dfdy(l, r) * go;
        }
        Ok(())
    }
}
