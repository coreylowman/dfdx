use crate::{
    shapes::{Shape, Unit},
    tensor::cpu::{Cpu, LendingIterator, StridedArray},
};

use super::{CmpKernel, EqKernelOp, GeKernelOp, GtKernelOp, LeKernelOp, LtKernelOp, NeKernelOp};

trait CmpOpCpuKernel<E: Unit> {
    fn func(lhs: E, rhs: E) -> bool;
}

impl<Op: CmpOpCpuKernel<E>, E: Unit> CmpKernel<Op, E> for Cpu {
    fn forward<S: Shape>(
        &self,
        lhs: &Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, bool>, Self::Err> {
        let mut out: Self::Storage<S, bool> = StridedArray::new(lhs.shape)?;
        let mut lhs_iter = lhs.iter();
        let mut rhs_iter = rhs.iter();
        let mut out_iter = out.iter_mut();
        while let Some((o, (l, r))) = out_iter.next().zip(lhs_iter.next().zip(rhs_iter.next())) {
            *o = Op::func(*l, *r);
        }
        Ok(out)
    }
}

impl<E: Unit> CmpOpCpuKernel<E> for EqKernelOp {
    fn func(lhs: E, rhs: E) -> bool {
        lhs == rhs
    }
}

impl<E: Unit> CmpOpCpuKernel<E> for NeKernelOp {
    fn func(lhs: E, rhs: E) -> bool {
        lhs != rhs
    }
}

impl<E: Unit> CmpOpCpuKernel<E> for GtKernelOp {
    fn func(lhs: E, rhs: E) -> bool {
        lhs > rhs
    }
}

impl<E: Unit> CmpOpCpuKernel<E> for GeKernelOp {
    fn func(lhs: E, rhs: E) -> bool {
        lhs >= rhs
    }
}

impl<E: Unit> CmpOpCpuKernel<E> for LtKernelOp {
    fn func(lhs: E, rhs: E) -> bool {
        lhs < rhs
    }
}

impl<E: Unit> CmpOpCpuKernel<E> for LeKernelOp {
    fn func(lhs: E, rhs: E) -> bool {
        lhs <= rhs
    }
}
