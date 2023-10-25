use crate::{
    shapes::{Shape, Unit},
    tensor::{
        cpu::{Cpu, LendingIterator},
        Tensor, ZerosTensor,
    },
};

use super::{
    CmpKernel, EqKernelOp, GeKernelOp, GtKernelOp, LeKernelOp, LtKernelOp, NeKernelOp,
    ScalarCmpKernel,
};

trait CmpOpCpuKernel<E: Unit> {
    fn func(lhs: E, rhs: E) -> bool;
}

impl<Op: CmpOpCpuKernel<E>, E: Unit> CmpKernel<Op, E> for Cpu {
    fn forward<S: Shape, T>(
        &self,
        lhs: &Tensor<S, E, Self, T>,
        rhs: &Tensor<S, E, Self, T>,
    ) -> Result<Tensor<S, bool, Self>, Self::Err> {
        let mut out: Tensor<S, bool, Self> = self.try_zeros_like(&lhs.shape)?;
        let mut lhs_iter = lhs.iter();
        let mut rhs_iter = rhs.iter();
        let mut out_iter = out.iter_mut();
        while let Some((o, (l, r))) = out_iter.next().zip(lhs_iter.next().zip(rhs_iter.next())) {
            *o = Op::func(*l, *r);
        }
        Ok(out)
    }
}

impl<Op: CmpOpCpuKernel<E>, E: Unit> ScalarCmpKernel<Op, E> for Cpu {
    fn forward<S: Shape, T>(
        &self,
        lhs: &Tensor<S, E, Self, T>,
        scalar: E,
    ) -> Result<Tensor<S, bool, Self>, Self::Err> {
        let mut out: Tensor<S, bool, Self> = self.try_zeros_like(&lhs.shape)?;
        let mut lhs_iter = lhs.iter();
        let mut out_iter = out.iter_mut();
        while let Some((o, l)) = out_iter.next().zip(lhs_iter.next()) {
            *o = Op::func(*l, scalar);
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
