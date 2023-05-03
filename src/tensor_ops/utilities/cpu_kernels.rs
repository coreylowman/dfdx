use std::borrow::Cow;

use super::ops::{BinaryKernel, UnaryKernel};
use crate::{
    shapes::{Dtype, Shape},
    tensor::{
        cpu::{Cpu, NdIndex},
        unique_id, Tensor, Tensorlike, ZerosTensor,
    },
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub trait UnaryDerivative<E>: Send + Sync {
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
        inp: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let mut out = match inp {
            Cow::Borrowed(inp) => {
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
            Cow::Owned(mut inp) => {
                // re-use the data buffer
                inp.id = unique_id();
                inp
            }
        };
        // NOTE: we can iterate over buf here because we know inp & out
        // have exact same strides due to clone.

        #[cfg(not(feature = "parallel"))]
        for x in out.buf_iter_mut() {
            *x = op.f(x);
        }

        #[cfg(feature = "parallel")]
        {
            let buf = std::sync::Arc::make_mut(&mut out.data);
            buf.data.par_iter_mut().for_each(|x| {
                *x = op.f(x);
            });
        }

        Ok(out)
    }
    fn backward<S: Shape>(
        &self,
        op: Op,
        inp: &impl Tensorlike<S, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &impl Tensorlike<S, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        match (inp.data(), out.data()) {
            (None, None) => {
                let df = op.const_df();
                #[cfg(not(feature = "parallel"))]
                for (i, x) in grad_inp.iter_mut().enumerate() {
                    *x += df * grad_out[i];
                }

                #[cfg(feature = "parallel")]
                grad_inp.par_iter_mut().enumerate().for_each(|(i, x)| {
                    *x += df * grad_out[i];
                });
            }
            (None, Some(out)) => {
                #[cfg(not(feature = "parallel"))]
                for (i, x) in grad_inp.iter_mut().enumerate() {
                    *x += op.df(&out[i]) * grad_out[i];
                }

                #[cfg(feature = "parallel")]
                grad_inp.par_iter_mut().enumerate().for_each(|(i, x)| {
                    *x += op.df(&out[i]) * grad_out[i];
                });
            }
            (Some(inp), None) => {
                #[cfg(not(feature = "parallel"))]
                for (i, x) in grad_inp.iter_mut().enumerate() {
                    *x += op.df(&inp[i]) * grad_out[i];
                }

                #[cfg(feature = "parallel")]
                grad_inp.par_iter_mut().enumerate().for_each(|(i, x)| {
                    *x += op.df(&inp[i]) * grad_out[i];
                });
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}

impl<E: Dtype, Op: BinaryDerivative<E> + Sync> BinaryKernel<Op, E> for Cpu {
    const BACKWARD_WITHOUT_DATA: bool = Op::HAS_CONST_DF;
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: Cow<Tensor<S, E, Self>>,
        rhs: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        match (lhs, rhs) {
            (Cow::Borrowed(lhs), Cow::Borrowed(rhs)) => {
                let mut out = self.try_zeros_like(&lhs.shape)?;

                #[cfg(not(feature = "parallel"))]
                {
                    let mut lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
                    let mut rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
                    for o in out.buf_iter_mut() {
                        let l = &lhs.data[lhs_idx.next().unwrap()];
                        let r = &rhs.data[rhs_idx.next().unwrap()];
                        *o = op.f(l, r);
                    }
                }

                #[cfg(feature = "parallel")]
                {
                    let lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
                    let rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
                    let buf = std::sync::Arc::make_mut(&mut out.data);
                    buf.data.par_iter_mut().enumerate().for_each(|(i, o)| {
                        let l = &lhs.data[lhs_idx.get_strided_index(i)];
                        let r = &rhs.data[rhs_idx.get_strided_index(i)];
                        *o = op.f(l, r);
                    });
                }
                Ok(out)
            }
            (Cow::Owned(mut lhs), Cow::Owned(mut rhs)) => {
                let lhs_valid = lhs.strides == lhs.shape.strides();
                let rhs_valid = rhs.strides == rhs.shape.strides();
                if lhs_valid || rhs_valid {
                    let lhs_count = std::sync::Arc::strong_count(&lhs.data);
                    let rhs_count = std::sync::Arc::strong_count(&rhs.data);
                    if rhs_valid && (rhs_count == 1 || !lhs_valid || lhs_count != 1) {
                        rhs.id = unique_id();
                        #[cfg(not(feature = "parallel"))]
                        {
                            let mut lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
                            for r in rhs.buf_iter_mut() {
                                *r = op.f(&lhs.data[lhs_idx.next().unwrap()], r);
                            }
                        }

                        #[cfg(feature = "parallel")]
                        {
                            let lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
                            let buf = std::sync::Arc::make_mut(&mut rhs.data);
                            buf.data.par_iter_mut().enumerate().for_each(|(i, r)| {
                                let l = &lhs.data[lhs_idx.get_strided_index(i)];
                                *r = op.f(l, r);
                            });
                        }
                        Ok(rhs)
                    } else {
                        lhs.id = unique_id();
                        #[cfg(not(feature = "parallel"))]
                        {
                            let mut rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
                            for l in lhs.buf_iter_mut() {
                                *l = op.f(l, &rhs.data[rhs_idx.next().unwrap()]);
                            }
                        }
                        #[cfg(feature = "parallel")]
                        {
                            let rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
                            let buf = std::sync::Arc::make_mut(&mut lhs.data);
                            buf.data.par_iter_mut().enumerate().for_each(|(i, l)| {
                                let r = &rhs.data[rhs_idx.get_strided_index(i)];
                                *l = op.f(l, r);
                            });
                        }
                        Ok(lhs)
                    }
                } else {
                    <Self as BinaryKernel<Op, E>>::forward(
                        self,
                        op,
                        Cow::Borrowed(&lhs),
                        Cow::Borrowed(&rhs),
                    )
                }
            }
            _ => unreachable!(),
        }
    }
    fn backward<S: Shape>(
        &self,
        op: Op,
        lhs: &impl Tensorlike<S, E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &impl Tensorlike<S, E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        #[cfg(not(feature = "parallel"))]
        match (lhs.data(), rhs.data()) {
            (Some(lhs_buf), Some(rhs_buf)) => {
                let mut lhs_idx = NdIndex::new(*lhs.shape(), lhs.strides());
                let mut rhs_idx = NdIndex::new(*rhs.shape(), rhs.strides());
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
            (None, None) => {
                assert!(Op::HAS_CONST_DF);
                let mut lhs_idx = NdIndex::new(*lhs.shape(), lhs.strides());
                let mut rhs_idx = NdIndex::new(*rhs.shape(), rhs.strides());
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

        #[cfg(feature = "parallel")]
        match (lhs.data(), rhs.data()) {
            (Some(lhs_buf), Some(rhs_buf)) => {
                let shape = *lhs.shape();
                let lhs_strides = lhs.strides();
                let rhs_strides = rhs.strides();
                let (lhs_idx, lhs_remap) = binary_bwd_permute::<S>(shape, lhs_strides);
                let (rhs_idx, rhs_remap) = binary_bwd_permute::<S>(shape, rhs_strides);
                let out_strides_for_lhs = permute::<S>(shape.strides(), lhs_remap);
                let rhs_strides_for_lhs = permute::<S>(rhs_strides, lhs_remap);
                let out_strides_for_rhs = permute::<S>(shape.strides(), rhs_remap);
                let lhs_strides_for_rhs = permute::<S>(lhs_strides, rhs_remap);
                let lhs_num_br = lhs.shape().num_elements() / grad_lhs.len();
                let rhs_num_br = rhs.shape().num_elements() / grad_rhs.len();
                rayon::join(
                    || {
                        grad_lhs.par_iter_mut().enumerate().for_each(|(i, gl)| {
                            let mut tmp = Default::default();
                            let l = &lhs_buf[i];
                            for j in 0..lhs_num_br {
                                let r = &rhs_buf
                                    [lhs_idx.restride(i * lhs_num_br + j, rhs_strides_for_lhs)];
                                let d = op.dfdx(l, r);
                                let go = grad_out
                                    [lhs_idx.restride(i * lhs_num_br + j, out_strides_for_lhs)];
                                tmp += d * go;
                            }
                            *gl += tmp;
                        });
                    },
                    || {
                        grad_rhs.par_iter_mut().enumerate().for_each(|(i, gr)| {
                            let mut tmp = Default::default();
                            let r = &rhs_buf[i];
                            for j in 0..rhs_num_br {
                                let l = &lhs_buf
                                    [rhs_idx.restride(i * rhs_num_br + j, lhs_strides_for_rhs)];
                                let d = op.dfdy(l, r);
                                let go = grad_out
                                    [rhs_idx.restride(i * rhs_num_br + j, out_strides_for_rhs)];
                                tmp += d * go;
                            }
                            *gr += tmp;
                        });
                    },
                );
            }
            (None, None) => {
                assert!(Op::HAS_CONST_DF);
                let shape = *lhs.shape();
                let (lhs_idx, lhs_remap) = binary_bwd_permute::<S>(shape, lhs.strides());
                let (rhs_idx, rhs_remap) = binary_bwd_permute::<S>(shape, rhs.strides());
                let out_strides_lhs = permute::<S>(shape.strides(), lhs_remap);
                let out_strides_rhs = permute::<S>(shape.strides(), rhs_remap);
                let dx = op.const_dfdx();
                let dy = op.const_dfdy();
                let lhs_num_br = lhs.shape().num_elements() / grad_lhs.len();
                let rhs_num_br = rhs.shape().num_elements() / grad_rhs.len();
                rayon::join(
                    || {
                        grad_lhs.par_iter_mut().enumerate().for_each(|(i, gl)| {
                            let mut tmp = Default::default();
                            for j in 0..lhs_num_br {
                                let out_i = lhs_idx.restride(i * lhs_num_br + j, out_strides_lhs);
                                tmp += dx * grad_out[out_i];
                            }
                            *gl += tmp;
                        });
                    },
                    || {
                        grad_rhs.par_iter_mut().enumerate().for_each(|(i, gr)| {
                            let mut tmp = Default::default();
                            for j in 0..rhs_num_br {
                                let out_i = rhs_idx.restride(i * rhs_num_br + j, out_strides_rhs);
                                tmp += dy * grad_out[out_i];
                            }
                            *gr += tmp;
                        });
                    },
                );
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}

#[inline(always)]
pub(crate) fn binary_bwd_permute<S: Shape>(
    shape: S,
    strides: S::Concrete,
) -> (NdIndex<S>, S::Concrete) {
    let dims = shape.concrete();
    let mut new_shape: S::Concrete = Default::default();
    let mut new_strides: S::Concrete = Default::default();
    let mut remap: S::Concrete = Default::default();
    let num_non_br_dims = strides.into_iter().filter(|&x| x != 0).count();

    let mut i_br = 0;
    let mut i_non_br = 0;
    for i_src in 0..S::NUM_DIMS {
        if strides[i_src] == 0 {
            // this axis is broadcasted
            new_shape[num_non_br_dims + i_br] = dims[i_src];
            new_strides[num_non_br_dims + i_br] = strides[i_src];
            remap[num_non_br_dims + i_br] = i_src;
            i_br += 1;
        } else {
            // this axis is not broadcasted
            new_shape[i_non_br] = dims[i_src];
            new_strides[i_non_br] = strides[i_src];
            remap[i_non_br] = i_src;
            i_non_br += 1;
        }
    }
    let idx = NdIndex {
        indices: Default::default(),
        shape: new_shape,
        strides: new_strides,
        next: Some(0),
        contiguous: (new_shape == dims && new_strides == shape.strides())
            .then(|| shape.num_elements()),
    };
    (idx, remap)
}

#[inline(always)]
pub(crate) fn permute<S: Shape>(strides: S::Concrete, remap: S::Concrete) -> S::Concrete {
    let mut new_strides: S::Concrete = Default::default();
    for i in 0..S::NUM_DIMS {
        new_strides[i] = strides[remap[i]];
    }
    new_strides
}
