use crate::{shapes::*, tensor::*};

use std::sync::Arc;

use num_traits::{Float, FromPrimitive};

fn make_4d<S: Shape>(strides: S::Concrete) -> [usize; 4] {
    match S::NUM_DIMS {
        3 => [0, strides[0], strides[1], strides[2]],
        4 => [strides[0], strides[1], strides[2], strides[3]],
        _ => panic!("Only implemented for 3d & 4d arrays"),
    }
}

impl super::Pool2DKind {
    fn init<E: Float>(&self) -> E {
        match self {
            super::Pool2DKind::Avg => E::zero(),
            super::Pool2DKind::Min => E::infinity(),
            super::Pool2DKind::Max => E::neg_infinity(),
        }
    }

    fn accum<E: Float>(&self, accum: &E, item: &E) -> E {
        match self {
            super::Pool2DKind::Avg => *accum + *item,
            super::Pool2DKind::Min => accum.min(*item),
            super::Pool2DKind::Max => accum.max(*item),
        }
    }

    fn normalize<E: Float + FromPrimitive>(&self, item: E, num_elements: usize) -> E {
        match self {
            super::Pool2DKind::Avg => item * E::from_f64(1.0 / num_elements as f64).unwrap(),
            super::Pool2DKind::Min => item,
            super::Pool2DKind::Max => item,
        }
    }

    fn filter<E: Float>(&self, item: E, needle: E, haystack: E) -> E {
        match self {
            super::Pool2DKind::Avg => item,
            super::Pool2DKind::Min => {
                if needle == haystack {
                    item
                } else {
                    E::zero()
                }
            }
            super::Pool2DKind::Max => {
                if needle == haystack {
                    item
                } else {
                    E::zero()
                }
            }
        }
    }
}

impl<E: Float + Dtype> super::Pool2DKernel<E> for Cpu {
    fn alloc<S: Shape>(&self, s: S) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_zeros_like(&s)
    }
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Tensor<I, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let buf = inp.data.as_ref();
        let out_buf = Arc::make_mut(&mut out.data);
        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let mut tmp = op.kind.init();
                        for k1 in 0..op.kernel {
                            let y = (oh * op.stride + op.dilation * k1).checked_sub(op.padding);
                            for k2 in 0..op.kernel {
                                let x = (ow * op.stride + op.dilation * k2).checked_sub(op.padding);
                                if let Some((y, x)) = y.zip(x) {
                                    if y < op.h_in && x < op.w_in {
                                        let inp_idx =
                                            b * istr[0] + c * istr[1] + y * istr[2] + x * istr[3];
                                        tmp = op.kind.accum(&tmp, &buf[inp_idx]);
                                    }
                                }
                            }
                        }
                        tmp = op.kind.normalize(tmp, op.kernel * op.kernel);
                        out_buf[b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3]] = tmp;
                    }
                }
            }
        }
        Ok(())
    }
    fn backward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Tensor<I, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let inp_buf = inp.data.as_ref();
        let out_buf = out.data.as_ref();

        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let out_idx = b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3];
                        let go = grad_out[out_idx];
                        let go = op.kind.normalize(go, op.kernel * op.kernel);
                        let vo = out_buf[out_idx];

                        for k1 in 0..op.kernel {
                            let y = (oh * op.stride + op.dilation * k1).checked_sub(op.padding);
                            for k2 in 0..op.kernel {
                                let x = (ow * op.stride + op.dilation * k2).checked_sub(op.padding);
                                if let Some((y, x)) = y.zip(x) {
                                    if x < op.w_in && y < op.h_in {
                                        let inp_idx =
                                            b * istr[0] + c * istr[1] + y * istr[2] + x * istr[3];
                                        grad_inp[inp_idx] +=
                                            op.kind.filter(go, inp_buf[inp_idx], vo);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
