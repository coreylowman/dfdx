use super::ops::{BinaryKernel, UnaryKernel};
use crate::{
    shapes::{Dtype, Shape},
    tensor::{
        cpu::{Cpu, LendingIterator, NdIndex},
        unique_id, GhostTensor, Tensor, ZerosTensor,
    },
};

pub trait UnaryDerivative<E> {
    /// Whether the [UnaryDerivative::df] function can re-use the output
    /// from [UnaryDerivative::f].
    const DF_USES_FX: bool;
    /// Whether the derivative of this op can be computed without
    /// any data.
    const HAS_CONST_DF: bool;

    fn f(&self, x: &E) -> E;

    /// Receives `f(x)` if [UnaryDerivative::DF_USES_FX] is true,
    /// otherwise `x`.
    fn df(&self, x: &E) -> E;

    fn const_df(&self) -> E {
        unimplemented!()
    }
}

pub trait BinaryDerivative<E>: std::fmt::Debug {
    /// Whether the derivative of this op can be computed without
    /// any data.
    const HAS_CONST_DF: bool;
    fn f(&self, x: &E, y: &E) -> E;
    fn dfdx(&self, x: &E, y: &E) -> E;
    fn dfdy(&self, x: &E, y: &E) -> E;
    fn const_dfdx(&self) -> E {
        unimplemented!()
    }
    fn const_dfdy(&self) -> E {
        unimplemented!()
    }
}

impl<E: Dtype, Op: UnaryDerivative<E>> UnaryKernel<Op, E> for Cpu {
    const BACKWARD_WITHOUT_INP: bool = Op::DF_USES_FX;
    const BACKWARD_WITHOUT_DATA: bool = Op::HAS_CONST_DF;

    fn forward<S: Shape>(
        &self,
        op: Op,
        inp: Result<&Tensor<S, E, Self>, Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let mut out = match inp {
            Ok(inp) => {
                // allocate a new data buffer
                Tensor {
                    id: unique_id(),
                    data: inp.data.clone(),
                    shape: inp.shape,
                    strides: inp.strides,
                    device: self.clone(),
                    tape: Default::default(),
                }
            }
            Err(mut inp) => {
                // re-use the data buffer
                inp.id = unique_id();
                inp
            }
        };
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
        inp: Result<&Tensor<S, E, Self>, &GhostTensor<S, E, Self>>,
        grad_inp: &mut Self::Vec<E>,
        out: Result<&Tensor<S, E, Self>, &GhostTensor<S, E, Self>>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        match (inp, out) {
            (Err(_), Err(_)) => {
                let df = op.const_df();
                for (i, x) in grad_inp.iter_mut().enumerate() {
                    *x += df * grad_out[i];
                }
            }
            (Err(_), Ok(out)) => {
                for (i, x) in grad_inp.iter_mut().enumerate() {
                    *x += op.df(&out.data[i]) * grad_out[i];
                }
            }
            (Ok(inp), Err(_)) => {
                for (i, x) in grad_inp.iter_mut().enumerate() {
                    *x += op.df(&inp.data[i]) * grad_out[i];
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}

impl<E: Dtype, Op: BinaryDerivative<E>> BinaryKernel<Op, E> for Cpu {
    const BACKWARD_WITHOUT_DATA: bool = Op::HAS_CONST_DF;
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: Result<&Tensor<S, E, Self>, Tensor<S, E, Self>>,
        rhs: Result<&Tensor<S, E, Self>, Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => {
                let mut out = self.try_zeros_like(&lhs.shape)?;
                let mut lhs_iter = lhs.iter();
                let mut rhs_iter = rhs.iter();
                for o in out.buf_iter_mut() {
                    let l = lhs_iter.next().unwrap();
                    let r = rhs_iter.next().unwrap();
                    *o = op.f(l, r);
                }
                Ok(out)
            }
            (Err(mut lhs), Err(mut rhs)) => {
                let lhs_valid = lhs.strides == lhs.shape.strides();
                let rhs_valid = rhs.strides == rhs.shape.strides();
                if lhs_valid || rhs_valid {
                    let lhs_count = std::sync::Arc::strong_count(&lhs.data);
                    let rhs_count = std::sync::Arc::strong_count(&rhs.data);
                    if lhs_valid && (lhs_count == 1 || !rhs_valid || rhs_count != 1) {
                        lhs.id = unique_id();
                        let mut rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
                        for l in lhs.buf_iter_mut() {
                            *l = op.f(l, &rhs.data[rhs_idx.next().unwrap()]);
                        }
                        Ok(lhs)
                    } else {
                        rhs.id = unique_id();
                        let mut lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
                        for r in rhs.buf_iter_mut() {
                            *r = op.f(&lhs.data[lhs_idx.next().unwrap()], r);
                        }
                        Ok(rhs)
                    }
                } else {
                    <Self as BinaryKernel<Op, E>>::forward(self, op, Ok(&lhs), Ok(&rhs))
                }
            }
            _ => unreachable!(),
        }
    }
    fn backward<S: Shape>(
        &self,
        op: Op,
        lhs: Result<&Tensor<S, E, Self>, &GhostTensor<S, E, Self>>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: Result<&Tensor<S, E, Self>, &GhostTensor<S, E, Self>>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => {
                let mut lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
                let mut rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
                let lhs_buf = lhs.data.as_ref();
                let rhs_buf = rhs.data.as_ref();
                // NOTE: we can use .buf_iter() here because we know the outcome of this op is
                // contiguous from forward
                for &go in grad_out.iter() {
                    let lhs_i = lhs_idx.next().unwrap();
                    let rhs_i = rhs_idx.next().unwrap();
                    let l = &lhs_buf[lhs_i];
                    let r = &rhs_buf[rhs_i];
                    grad_lhs[lhs_i] += op.dfdx(l, r) * go;
                    grad_rhs[rhs_i] += op.dfdy(l, r) * go;
                }
            }
            (Err(lhs), Err(rhs)) => {
                assert!(Op::HAS_CONST_DF);
                let mut lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
                let mut rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
                let dx = op.const_dfdx();
                let dy = op.const_dfdy();
                for &go in grad_out.iter() {
                    let lhs_i = lhs_idx.next().unwrap();
                    let rhs_i = rhs_idx.next().unwrap();
                    grad_lhs[lhs_i] += dx * go;
                    grad_rhs[rhs_i] += dy * go;
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}
